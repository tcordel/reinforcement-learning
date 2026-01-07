import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from pettingzoo.classic import connect_four_v3


# --------------------------
# Model: Policy + Value net
# --------------------------

class ResBlock(nn.Module):
    def __init__(self, c: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, c)

    def forward(self, x):
        r = x
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return F.relu(x + r)


class PVNet(nn.Module):
    """
    Input:  (B, 2, 6, 7)
    Output: policy_logits (B, 7), value (B,)
    """
    def __init__(self, channels: int = 64, n_blocks: int = 8, gn_groups: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels, gn_groups) for _ in range(n_blocks)])

        # Policy head
        self.pi_conv = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.GroupNorm(1, 2),
            nn.ReLU(),
        )
        self.pi_fc = nn.Linear(2 * 6 * 7, 7)

        # Value head
        self.v_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.GroupNorm(1, 1),
            nn.ReLU(),
        )
        self.v_fc1 = nn.Linear(1 * 6 * 7, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)

        p = self.pi_conv(x).flatten(1)
        logits = self.pi_fc(p)

        v = self.v_conv(x).flatten(1)
        v = F.relu(self.v_fc1(v))
        value = torch.tanh(self.v_fc2(v)).squeeze(-1)

        return logits, value


# --------------------------
# Helpers
# --------------------------

def obs_to_tensor(obs: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PettingZoo connect_four_v3:
      obs["observation"] shape (6,7,2) with planes already "to-play" then opponent
      obs["action_mask"] shape (7,)
    Returns:
      x: (1,2,6,7) float32
      mask: (1,7) bool
    """
    board = obs["observation"]  # (6,7,2)
    mask = obs["action_mask"]   # (7,)
    x = torch.from_numpy(board).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    m = torch.from_numpy(mask).to(device=device, dtype=torch.bool).unsqueeze(0)
    return x, m


def mask_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    # Use a large negative number instead of -inf to avoid NaNs
    neg = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
    return torch.where(legal_mask, logits, neg)


def forward_mirror_invariant(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enforce horizontal symmetry in forward pass:
      logits(x) averaged with unflipped logits(flip(x))
      value averaged with value(flip(x))
    x: (B,2,6,7)
    """
    logits1, v1 = model(x)
    xf = x.flip(-1)
    logits2, v2 = model(xf)
    logits2 = logits2.flip(-1)
    return 0.5 * (logits1 + logits2), 0.5 * (v1 + v2)


@dataclass
class Step:
    state: torch.Tensor      # (2,6,7) on CPU
    mask: torch.Tensor       # (7,) bool on CPU
    action: int
    logp_old: float
    value_old: float
    player_id: str           # "player_0" or "player_1"


# --------------------------
# Rollout (self-play)
# --------------------------

@torch.no_grad()
def play_one_game(
    env,
    model: nn.Module,
    device: torch.device,
    temperature: float = 1.0,
    use_mirror_invariance: bool = True,
) -> Tuple[List[Step], Dict[str, float]]:
    """
    Plays one full Connect4 game in AEC PettingZoo.
    Returns:
      steps: chronological list of decisions
      final_rewards: dict agent_id -> final reward (+1/-1/0)
    """
    env.reset()
    steps: List[Step] = []

    final_rewards = {}
    while env.agents:
        agent = env.agent_selection
        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            final_rewards[agent] = reward
            env.step(None)
            continue

        x, m = obs_to_tensor(obs, device)

        if use_mirror_invariance:
            logits, value = forward_mirror_invariant(model, x)
        else:
            logits, value = model(x)

        # temperature (for exploration): logits / T
        logits = logits / max(temperature, 1e-6)

        logits = mask_logits(logits, m)
        dist = Categorical(logits=logits)

        action = int(dist.sample().item())
        logp = float(dist.log_prob(torch.tensor(action, device=device)).item())
        v = float(value.item())

        # store on CPU to keep rollout cheap
        steps.append(
            Step(
                state=x.squeeze(0).cpu(),
                mask=m.squeeze(0).cpu(),
                action=action,
                logp_old=logp,
                value_old=v,
                player_id=agent,
            )
        )

        env.step(action)

    return steps, final_rewards


# --------------------------
# PPO Update
# --------------------------

def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_steps: List[Step],
    final_rewards_per_game: List[Dict[str, float]],
    game_slices: List[Tuple[int, int]],  # for each game: [start, end) indices in batch_steps
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    epochs: int = 4,
    minibatch_size: int = 256,
    use_mirror_invariance: bool = True,
):
    """
    We compute returns as final outcome from the acting player's perspective:
      return_t = final_reward[player_id]
    Advantage:
      adv_t = return_t - value_old
    """
    # Build tensors
    n = len(batch_steps)
    states = torch.stack([s.state for s in batch_steps], dim=0).to(device)          # (N,2,6,7)
    masks  = torch.stack([s.mask for s in batch_steps], dim=0).to(device)           # (N,7)
    actions = torch.tensor([s.action for s in batch_steps], device=device)          # (N,)
    logp_old = torch.tensor([s.logp_old for s in batch_steps], device=device)       # (N,)
    value_old = torch.tensor([s.value_old for s in batch_steps], device=device)     # (N,)

    # returns: fill by looking up each step's game final reward for that player
    returns = torch.empty((n,), device=device, dtype=torch.float32)
    for gi, (a, b) in enumerate(game_slices):
        fr = final_rewards_per_game[gi]
        for idx in range(a, b):
            pid = batch_steps[idx].player_id
            returns[idx] = float(fr[pid])

    advantages = returns - value_old
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    model.train()
    idxs = torch.arange(n, device=device)

    for _ in range(epochs):
        perm = idxs[torch.randperm(n)]
        for start in range(0, n, minibatch_size):
            mb = perm[start:start + minibatch_size]

            x = states[mb]
            m = masks[mb]
            a = actions[mb]
            old_lp = logp_old[mb]
            ret = returns[mb]
            adv = advantages[mb]

            if use_mirror_invariance:
                logits, v = forward_mirror_invariant(model, x)
            else:
                logits, v = model(x)

            logits = mask_logits(logits, m)
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(a)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lp - old_lp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (ret - v).pow(2).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


# --------------------------
# Evaluation vs Random
# --------------------------

@torch.no_grad()
def eval_vs_random(model: nn.Module, device: torch.device, n_games: int = 50) -> Dict[str, float]:
    """
    Play games where player_0 uses model (greedy), player_1 is random.
    Returns win/draw/loss rates for player_0.
    """
    env = connect_four_v3.env()
    wins = draws = losses = 0

    player = None
    for _ in range(n_games):
        env.reset()
        final_rewards = {}
        player = "player_0" if player is None or player == "player_1" else "player_1"
        while env.agents:
            agent = env.agent_selection
            obs, reward, term, trunc, info = env.last()
            if term or trunc:
                final_rewards[agent] = reward
                env.step(None)
                continue

            mask = obs["action_mask"]
            legal = np.flatnonzero(mask)

            if agent == player:
                x, m = obs_to_tensor(obs, device)
                logits, _ = forward_mirror_invariant(model, x)
                logits = mask_logits(logits, m)
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                action = int(np.random.choice(legal))

            env.step(action)

        r0 = float(final_rewards[player])
        if r0 > 0:
            wins += 1
        elif r0 < 0:
            losses += 1
        else:
            draws += 1

    return {
        "win_rate": wins / n_games,
        "draw_rate": draws / n_games,
        "loss_rate": losses / n_games,
    }


# --------------------------
# Main training loop
# --------------------------

device_str= "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
def train(
    total_updates: int = 2000,
    games_per_update: int = 32,
    temperature_start: float = 1.5,
    temperature_end: float = 0.6,
):

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = PVNet(channels=64, n_blocks=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    env = connect_four_v3.env()

    for upd in range(1, total_updates + 1):
        # Temperature schedule (linear)
        t = (upd - 1) / max(total_updates - 1, 1)
        temperature = temperature_start + t * (temperature_end - temperature_start)

        batch_steps: List[Step] = []
        final_rewards_per_game: List[Dict[str, float]] = []
        game_slices: List[Tuple[int, int]] = []

        # Collect rollouts
        for _ in range(games_per_update):
            start = len(batch_steps)
            steps, final_rewards = play_one_game(
                env, model, device,
                temperature=temperature,
                use_mirror_invariance=True,
            )
            batch_steps.extend(steps)
            final_rewards_per_game.append(final_rewards)
            end = len(batch_steps)
            game_slices.append((start, end))

        # PPO update
        ppo_update(
            model=model,
            optimizer=optimizer,
            device=device,
            batch_steps=batch_steps,
            final_rewards_per_game=final_rewards_per_game,
            game_slices=game_slices,
            clip_eps=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            epochs=4,
            minibatch_size=256,
            use_mirror_invariance=True,
        )

        print(upd)
        # Evaluate sometimes
        if upd == 1 or upd % 500 == 0:
            stats = eval_vs_random(model, device, n_games=50)
            print(f"[upd={upd:04d}] steps={len(batch_steps):5d} temp={temperature:.3f} "
                  f"win/draw/loss={stats['win_rate']:.2f}/{stats['draw_rate']:.2f}/{stats['loss_rate']:.2f}")

    return model


if __name__ == "__main__":


    model = train(total_updates=10000)

    torch.save(model.state_dict(), "./vibe.pth")

    env_manual = connect_four_v3.env(render_mode="human")
    player = None
    while True:
        env_manual.reset(seed=42)
        player = "player_0" if player is None or player == "player_1" else "player_1"
        for agent in env_manual.agent_iter():
            observation, reward, termination, truncation, info = env_manual.last()

            learning_model_round = agent == player
            if termination or truncation:
                action = None
            else:
                if learning_model_round:
                    x, m = obs_to_tensor(observation, device)
                    logits, _ = forward_mirror_invariant(model, x)
                    logits = mask_logits(logits, m)
                    action = int(torch.argmax(logits, dim=-1).item())
                else:
                    ok = False
                    while not ok:
                        ok = True
                        try:
                            action = input()
                            if action == "q":
                                break
                            action = np.array(action, dtype=np.int16)
                        except:
                            ok = False

            env_manual.step(action)
