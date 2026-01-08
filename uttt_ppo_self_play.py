# uttt_ppo_selfplay.py
# Python 3.10+
# Points importants (pour éviter “ça tourne en rond”)
# Transition correcte (single-agent view)
# On ne stocke des transitions que quand c’est le tour de la policy en apprentissage, et on “absorbe” le coup adverse pour construire un vrai MDP induit.
# ➡️ Ça évite les bugs GAE classiques en self-play multi-agent.
# 
# Masque d’action partout (policy + sampling + dist)
# UTTT sans mask strict = apprentissage incohérent.
# 
# Métriques qui disent vraiment quelque chose (UTTT est illisible à l’œil)
# 
# eval/vs_random_win : sanity check
# 
# eval/vs_snapshot_win + Elo : progrès relatif
# 
# train/approx_kl, entropy, clipfrac : stabilité PPO
# 
# explained_variance : le critic apprend-il vraiment ?
# 
# Si tu veux que ça apprenne plus vite (sans “tricher” MCTS)
# 
# Deux leviers très efficaces, que je n’ai pas activés par défaut :
# 
# micro_win_reward léger (ex: 0.05) → ça densifie le signal (mais ça “shape” le jeu).
# 
# curriculum : d’abord p_vs_random=1.0 puis descendre vers 0.2.
# 
# Si tu me dis tes ressources (CPU/GPU) et ton objectif (jouer “bien” vs humain ou simplement converger), je te donne des hyperparams ciblés + un protocole Elo “league” plus rigoureux.
import random
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard.writer import SummaryWriter


# ============================================================
# 1) ENVIRONNEMENT UTTT (actions = 81 cases, ordre row-major 9x9)
# ============================================================

def _check_3x3_win(grid3: np.ndarray, player: int) -> bool:
    # grid3 shape (3,3), values in {-1,0,1}
    # win if any row/col/diag all equal player
    p = player
    if (grid3[0, :] == p).all() or (grid3[1, :] == p).all() or (grid3[2, :] == p).all():
        return True
    if (grid3[:, 0] == p).all() or (grid3[:, 1] == p).all() or (grid3[:, 2] == p).all():
        return True
    if (np.diag(grid3) == p).all() or (np.diag(np.fliplr(grid3)) == p).all():
        return True
    return False


def _check_macro_win(micro_status: np.ndarray, player: int) -> bool:
    # micro_status length 9: 0 ongoing, 1 X won, -1 O won, 2 draw
    # macro win only counts boards won by player; draws count as empty (0)
    ms = micro_status.copy()
    ms[ms == 2] = 0
    ms = ms.reshape(3, 3)
    p = player
    if (ms[0, :] == p).all() or (ms[1, :] == p).all() or (ms[2, :] == p).all():
        return True
    if (ms[:, 0] == p).all() or (ms[:, 1] == p).all() or (ms[:, 2] == p).all():
        return True
    if (np.diag(ms) == p).all() or (np.diag(np.fliplr(ms)) == p).all():
        return True
    return False


class UTTTEnv:
    """
    Ultimate Tic-Tac-Toe.
    - board: 9x9, values in {-1,0,1}
    - micro_status: 9, values in {0 ongoing, 1 X won, -1 O won, 2 draw}
    - current_player: 1 (X) or -1 (O)
    - next_board: -1 (any) or 0..8 (constrained microboard index)
    Action space: 81 (global cell index = r*9+c).
    """

    def __init__(self, micro_win_reward: float = 0.0):
        # micro_win_reward: shaping option (0.0 = jeu pur)
        self.micro_win_reward = float(micro_win_reward)
        self.rng = np.random.default_rng()

        self.board = np.zeros((9, 9), dtype=np.int8)
        self.micro_status = np.zeros((9,), dtype=np.int8)
        self.current_player = 1
        self.next_board = -1
        self.done = False
        self.winner: Optional[int] = None  # 1 / -1 / 0 (draw) / None ongoing

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> Dict:
        if seed is not None:
            self.seed(seed)

        self.board.fill(0)
        self.micro_status.fill(0)
        self.current_player = 1 if self.rng.integers(0, 2) == 0 else -1
        self.next_board = -1
        self.done = False
        self.winner = None
        return self._get_obs()

    def _legal_mask(self) -> np.ndarray:
        mask = np.zeros((81,), dtype=np.bool_)
        if self.done:
            return mask

        # Determine which microboards are allowed
        if self.next_board != -1 and self.micro_status[self.next_board] == 0:
            allowed = [self.next_board]
        else:
            allowed = [i for i in range(9) if self.micro_status[i] == 0]

        for mb in allowed:
            mr, mc = divmod(mb, 3)  # microboard position in macro grid
            r0, c0 = mr * 3, mc * 3
            sub = self.board[r0:r0+3, c0:c0+3]
            empties = np.argwhere(sub == 0)
            for (lr, lc) in empties:
                gr, gc = r0 + lr, c0 + lc
                a = gr * 9 + gc
                mask[a] = True
        return mask

    def _action_to_coords(self, action: int) -> Tuple[int, int]:
        r = action // 9
        c = action % 9
        return r, c

    def _coords_to_micro_local(self, r: int, c: int) -> Tuple[int, int]:
        # micro index (0..8), local cell index (0..8)
        mb = (r // 3) * 3 + (c // 3)
        lr = r % 3
        lc = c % 3
        local = lr * 3 + lc
        return mb, local

    def _update_micro_status(self, mb: int, player: int) -> float:
        # returns shaping reward for this step
        if self.micro_status[mb] != 0:
            return 0.0

        mr, mc = divmod(mb, 3)
        r0, c0 = mr * 3, mc * 3
        sub = self.board[r0:r0+3, c0:c0+3]

        if _check_3x3_win(sub, player):
            self.micro_status[mb] = player
            return self.micro_win_reward
        if not (sub == 0).any():
            self.micro_status[mb] = 2  # draw
        return 0.0

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Executes action for current_player, toggles current_player, returns obs for next player.
        Reward returned here is *shaping only* + (terminal outcome for the player who acted) if terminal on this move.
        For PPO, we will still use final outcome properly from env.winner at episode end.
        """
        if self.done:
            return self._get_obs(), 0.0, True, {}

        legal = self._legal_mask()
        if action < 0 or action >= 81 or not legal[action]:
            # illegal move => immediate loss for current_player
            self.done = True
            self.winner = -self.current_player
            # reward for player who acted (illegal) is -1
            r = -1.0
            # toggle player for consistency
            self.current_player *= -1
            return self._get_obs(), r, True, {"illegal": True}

        r, c = self._action_to_coords(action)
        self.board[r, c] = self.current_player

        mb, local = self._coords_to_micro_local(r, c)
        shaping = self._update_micro_status(mb, self.current_player)

        # Check macro win/draw
        if _check_macro_win(self.micro_status, self.current_player):
            self.done = True
            self.winner = self.current_player
        elif np.all(self.micro_status != 0):
            self.done = True
            self.winner = 0
        else:
            self.done = False
            self.winner = None

        # next_board is determined by local cell inside the microboard we played
        # BUT if that target microboard is already finished, next player can play anywhere.
        target = local
        if target != -1 and self.micro_status[target] == 0:
            self.next_board = target
        else:
            self.next_board = -1

        # Terminal reward for the mover (optional; PPO code below uses env.winner anyway)
        terminal_r = 0.0
        if self.done:
            if self.winner == self.current_player:
                terminal_r = 1.0
            elif self.winner == 0:
                terminal_r = 0.0
            else:
                terminal_r = -1.0  # should not happen here except illegal handled above

        # toggle player
        self.current_player *= -1
        return self._get_obs(), float(shaping + terminal_r), self.done, {}

    def _get_obs(self) -> Dict:
        """
        Observation is always from perspective of current_player ("to-play canonical").
        Returns:
          obs: float32 array (C,9,9)
          action_mask: bool array (81,)
        """
        cp = self.current_player
        board = self.board

        # Planes for pieces
        me = (board == cp).astype(np.float32)
        opp = (board == -cp).astype(np.float32)

        # Legal moves plane
        mask = self._legal_mask()
        legal_plane = np.zeros((9, 9), dtype=np.float32)
        if mask.any():
            idxs = np.flatnonzero(mask)
            rs = idxs // 9
            cs = idxs % 9
            legal_plane[rs, cs] = 1.0

        # Micro status planes expanded to 9x9
        micro_me = np.zeros((9, 9), dtype=np.float32)
        micro_opp = np.zeros((9, 9), dtype=np.float32)
        micro_draw = np.zeros((9, 9), dtype=np.float32)
        for mb in range(9):
            mr, mc = divmod(mb, 3)
            r0, c0 = mr * 3, mc * 3
            st = self.micro_status[mb]
            if st == cp:
                micro_me[r0:r0+3, c0:c0+3] = 1.0
            elif st == -cp:
                micro_opp[r0:r0+3, c0:c0+3] = 1.0
            elif st == 2:
                micro_draw[r0:r0+3, c0:c0+3] = 1.0

        # Next-board constraint plane
        next_plane = np.zeros((9, 9), dtype=np.float32)
        if self.next_board == -1:
            next_plane[:, :] = 1.0
        else:
            mr, mc = divmod(self.next_board, 3)
            r0, c0 = mr * 3, mc * 3
            next_plane[r0:r0+3, c0:c0+3] = 1.0

        obs = np.stack([me, opp, legal_plane, micro_me, micro_opp, micro_draw, next_plane], axis=0)
        return {"obs": obs.astype(np.float32), "action_mask": mask, "current_player": cp, "winner": self.winner}


# ============================================================
# 2) RÉSEAU POLICY/VALUE (conv résiduel, policy logits=81)
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, c: int, gn_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(gn_groups, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(gn_groups, c)

    def forward(self, x):
        r = x
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return F.relu(x + r)


class UTTTPVNet(nn.Module):
    """
    Input: (B, C=7, 9, 9)
    Output:
      logits: (B, 81)
      value:  (B,)
    """

    def __init__(self, in_channels: int = 7, channels: int = 64, n_blocks: int = 8, gn_groups: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels, gn_groups) for _ in range(n_blocks)])

        # Policy head -> heatmap 9x9 then flatten to 81 logits (aligné avec action space)
        self.pi_head = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.GroupNorm(1, 16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1, bias=True),
        )

        # Value head
        self.v_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            nn.GroupNorm(1, 8),
            nn.ReLU(),
        )
        self.v_fc1 = nn.Linear(8 * 9 * 9, 128)
        self.v_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.blocks(x)

        pi = self.pi_head(x)              # (B,1,9,9)
        logits = pi.flatten(1)            # (B,81)

        v = self.v_head(x).flatten(1)     # (B,8*9*9)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v)).squeeze(-1)
        return logits, v


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # mask: bool (B,81)
    neg = logits.new_full((), -1e9)
    return torch.where(mask, logits, neg)


# ============================================================
# 3) PPO + GAE (single-agent view) + self-play vs snapshot pool
# ============================================================

@dataclass
class Transition:
    obs: torch.Tensor        # (C,9,9) CPU
    mask: torch.Tensor       # (81,) bool CPU
    action: int
    logp: float
    value: float
    reward: float
    done: bool
    next_obs: torch.Tensor   # (C,9,9) CPU
    next_mask: torch.Tensor  # (81,) bool CPU


def obs_to_torch(o: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(o["obs"]).to(device=device, dtype=torch.float32)  # (C,9,9)
    m = torch.from_numpy(o["action_mask"]).to(device=device, dtype=torch.bool)  # (81,)
    return x, m


@torch.no_grad()
def act(model: nn.Module, x: torch.Tensor, m: torch.Tensor, temperature: float) -> Tuple[int, float, float]:
    # x: (C,9,9) on device; m: (81,)
    logits, v = model(x.unsqueeze(0))    # (1,81), (1,)
    logits = logits / max(temperature, 1e-6)
    logits = masked_logits(logits, m.unsqueeze(0))
    dist = Categorical(logits=logits)
    a = int(dist.sample().item())
    logp = float(dist.log_prob(torch.tensor(a, device=x.device)).item())
    return a, logp, float(v.item())


@torch.no_grad()
def act_random(mask: np.ndarray, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal))


def collect_rollouts(
    model: nn.Module,
    opponent_pool: List[nn.Module],
    device: torch.device,
    rollout_steps: int = 4096,
    n_envs: int = 16,
    temperature: float = 1.0,
    p_vs_random: float = 0.2,
) -> Tuple[List[Transition], Dict[str, float]]:
    """
    Collect rollouts from the perspective of the learning policy only.
    Opponent is either random or a snapshot from opponent_pool.
    Each env episode: learning side is randomly assigned to X or O.
    """
    envs = [UTTTEnv(micro_win_reward=0.0) for _ in range(n_envs)]
    rng = np.random.default_rng()

    transitions: List[Transition] = []
    stats = {"episodes": 0.0, "wins": 0.0, "losses": 0.0, "draws": 0.0}

    # init
    env_state = []
    for i, env in enumerate(envs):
        o = env.reset(seed=int(rng.integers(0, 10_000_000)))
        env_state.append(o)

    while len(transitions) < rollout_steps:
        for i, env in enumerate(envs):
            if len(transitions) >= rollout_steps:
                break

            # if env already done, start a new episode
            if env.done:
                # tally outcome for the learning side? we track per-episode at the moment it ends (below)
                o = env.reset(seed=int(rng.integers(0, 10_000_000)))
                env_state[i] = o

            # Sample opponent type and assign learning sign at episode start
            # We store these in env object attributes for convenience (python allows dynamic fields)
            if not hasattr(env, "learning_sign") or getattr(env, "_episode_fresh", True):
                env.learning_sign = 1 if rng.integers(0, 2) == 0 else -1   # learning plays X(1) or O(-1)
                env.opponent_is_random = (rng.random() < p_vs_random)
                if env.opponent_is_random:
                    env.opponent_model = None
                else:
                    env.opponent_model = random.choice(opponent_pool)
                env._episode_fresh = False

            # Play until it's learning policy's turn OR terminal,
            # but since env_state is always current_player-to-play obs,
            # we can check env.current_player.
            # If it's opponent's turn, we let opponent act.
            while (not env.done) and (env.current_player != env.learning_sign):
                o = env_state[i]
                if env.opponent_is_random:
                    a_op = act_random(o["action_mask"], rng)
                else:
                    x_op, m_op = obs_to_torch(o, device)
                    a_op, _, _ = act(env.opponent_model, x_op, m_op, temperature=1.0)  # opponent temp=1
                o2, _, done, info = env.step(a_op)
                env_state[i] = o2

            if env.done:
                # episode ended on opponent move (possible)
                outcome = env.winner if env.winner is not None else 0
                score = float(outcome * env.learning_sign)  # +1 win, -1 loss, 0 draw for learning side
                stats["episodes"] += 1.0
                stats["wins"] += 1.0 if score > 0 else 0.0
                stats["losses"] += 1.0 if score < 0 else 0.0
                stats["draws"] += 1.0 if score == 0 else 0.0
                env._episode_fresh = True
                continue

            # Now it's learning policy's turn (env.current_player == env.learning_sign)
            o = env_state[i]
            x, m = obs_to_torch(o, device)

            a, logp, v = act(model, x, m, temperature=temperature)

            # Step with learning action
            o2, shaping_r, done, info = env.step(a)
            env_state[i] = o2

            # After learning moves, we advance opponent until next learning turn or terminal,
            # so that (s -> s_next) is a correct transition in the induced MDP.
            o_mid = env_state[i]
            while (not env.done) and (env.current_player != env.learning_sign):
                if env.opponent_is_random:
                    a_op = act_random(o_mid["action_mask"], rng)
                else:
                    x_op, m_op = obs_to_torch(o_mid, device)
                    a_op, _, _ = act(env.opponent_model, x_op, m_op, temperature=1.0)
                o_mid, _, _, _ = env.step(a_op)
                env_state[i] = o_mid

            # Build transition reward:
            # - shaping reward from our move (optionnel; micro_win_reward=0 by default)
            # - if terminal occurred before next learning turn, give final outcome here
            final_r = 0.0
            done_flag = env.done
            if done_flag:
                outcome = env.winner if env.winner is not None else 0
                final_r = float(outcome * env.learning_sign)
                stats["episodes"] += 1.0
                stats["wins"] += 1.0 if final_r > 0 else 0.0
                stats["losses"] += 1.0 if final_r < 0 else 0.0
                stats["draws"] += 1.0 if final_r == 0 else 0.0
                env._episode_fresh = True

            total_r = float(shaping_r + final_r)

            # next state is either next learning turn obs or terminal obs (we still store obs for bootstrap)
            o_next = env_state[i]
            nx = torch.from_numpy(o_next["obs"]).to(dtype=torch.float32).cpu()
            nm = torch.from_numpy(o_next["action_mask"]).to(dtype=torch.bool).cpu()

            transitions.append(
                Transition(
                    obs=torch.from_numpy(o["obs"]).to(dtype=torch.float32).cpu(),
                    mask=torch.from_numpy(o["action_mask"]).to(dtype=torch.bool).cpu(),
                    action=a,
                    logp=logp,
                    value=v,
                    reward=total_r,
                    done=done_flag,
                    next_obs=nx,
                    next_mask=nm,
                )
            )

    # Convert episode stats to rates
    eps = max(stats["episodes"], 1.0)
    stats["win_rate"] = stats["wins"] / eps
    stats["loss_rate"] = stats["losses"] / eps
    stats["draw_rate"] = stats["draws"] / eps
    return transitions, stats


def compute_gae(
    transitions: List[Transition],
    model: nn.Module,
    device: torch.device,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns tensors on device:
      obs (N,C,9,9), masks (N,81), actions (N,), logp_old (N,), returns (N,), adv (N,)
    """
    N = len(transitions)
    obs = torch.stack([t.obs for t in transitions], dim=0).to(device)              # (N,C,9,9)
    masks = torch.stack([t.mask for t in transitions], dim=0).to(device)           # (N,81)
    actions = torch.tensor([t.action for t in transitions], device=device)         # (N,)
    logp_old = torch.tensor([t.logp for t in transitions], device=device)          # (N,)
    values = torch.tensor([t.value for t in transitions], device=device)           # (N,)
    rewards = torch.tensor([t.reward for t in transitions], device=device)         # (N,)
    dones = torch.tensor([t.done for t in transitions], device=device, dtype=torch.float32)  # (N,)

    # Bootstrap values for next states
    with torch.no_grad():
        next_obs = torch.stack([t.next_obs for t in transitions], dim=0).to(device)
        next_masks = torch.stack([t.next_mask for t in transitions], dim=0).to(device)
        next_logits, next_values = model(next_obs)
        # note: next_values shape (N,)
        next_values = next_values

    # GAE in reverse
    adv = torch.zeros((N,), device=device)
    gae = 0.0
    for t in reversed(range(N)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nonterminal * next_values[t] - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae

    returns = adv + values
    return obs, masks, actions, logp_old, returns.detach(), adv.detach()


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    # 1 - Var[y-yp]/Var[y]
    var_y = torch.var(y_true)
    if var_y.item() < 1e-8:
        return float("nan")
    return float(1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8))


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    masks: torch.Tensor,
    actions: torch.Tensor,
    logp_old: torch.Tensor,
    returns: torch.Tensor,
    adv: torch.Tensor,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    epochs: int = 4,
    minibatch_size: int = 512,
    value_clip: float = 0.2,
) -> Dict[str, float]:
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    # metrics accum
    total_pi_loss = 0.0
    total_v_loss = 0.0
    total_ent = 0.0
    total_kl = 0.0
    total_clipfrac = 0.0
    n_updates = 0

    # cache old values for value clipping
    with torch.no_grad():
        _, v_old = model(obs)
        v_old = v_old.detach()

    idxs = torch.arange(N, device=obs.device)
    for _ in range(epochs):
        perm = idxs[torch.randperm(N)]
        for start in range(0, N, minibatch_size):
            mb = perm[start:start + minibatch_size]

            logits, v = model(obs[mb])
            logits = masked_logits(logits, masks[mb])
            dist = Categorical(logits=logits)

            new_logp = dist.log_prob(actions[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv[mb]
            pi_loss = -torch.min(surr1, surr2).mean()

            # Value clipping (PPO2 style)
            v_clipped = v_old[mb] + torch.clamp(v - v_old[mb], -value_clip, value_clip)
            v_loss1 = (v - returns[mb]).pow(2)
            v_loss2 = (v_clipped - returns[mb]).pow(2)
            v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

            loss = pi_loss + vf_coef * v_loss - ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (logp_old[mb] - new_logp).mean()
                clipfrac = (torch.abs(ratio - 1.0) > clip_eps).float().mean()

            total_pi_loss += float(pi_loss.item())
            total_v_loss += float(v_loss.item())
            total_ent += float(entropy.item())
            total_kl += float(approx_kl.item())
            total_clipfrac += float(clipfrac.item())
            n_updates += 1

    # final metrics
    with torch.no_grad():
        _, v_pred = model(obs)
    ev = explained_variance(v_pred, returns)

    return {
        "pi_loss": total_pi_loss / max(n_updates, 1),
        "v_loss": total_v_loss / max(n_updates, 1),
        "entropy": total_ent / max(n_updates, 1),
        "approx_kl": total_kl / max(n_updates, 1),
        "clipfrac": total_clipfrac / max(n_updates, 1),
        "explained_variance": ev,
        "adv_mean": float(adv.mean().item()),
        "adv_std": float(adv.std(unbiased=False).item()),
        "ret_mean": float(returns.mean().item()),
        "ret_std": float(returns.std(unbiased=False).item()),
    }


# ============================================================
# 4) ÉVALUATION (vs random, vs snapshot) + Elo simple
# ============================================================

def play_match(
    policy_a: nn.Module,
    policy_b: nn.Module,
    device: torch.device,
    n_games: int = 50,
    temperature: float = 0.1,
) -> Dict[str, float]:
    """
    policy_a joue contre policy_b. À chaque game, A est assigné aléatoirement à X ou O.
    Retourne win/draw/loss de A.
    """
    rng = np.random.default_rng()
    env = UTTTEnv(micro_win_reward=0.0)
    wins = draws = losses = 0

    for _ in range(n_games):
        o = env.reset(seed=int(rng.integers(0, 10_000_000)))
        a_sign = 1 if rng.integers(0, 2) == 0 else -1  # policy_a is X or O

        while not env.done:
            if env.current_player == a_sign:
                x, m = obs_to_torch(o, device)
                a, _, _ = act(policy_a, x, m, temperature=temperature)
            else:
                x, m = obs_to_torch(o, device)
                a, _, _ = act(policy_b, x, m, temperature=temperature)
            o, _, _, _ = env.step(a)

        outcome = env.winner if env.winner is not None else 0
        score = outcome * a_sign
        if score > 0:
            wins += 1
        elif score < 0:
            losses += 1
        else:
            draws += 1

    return {"win_rate": wins / n_games, "draw_rate": draws / n_games, "loss_rate": losses / n_games}


class Elo:
    def __init__(self, k: float = 16.0):
        self.k = float(k)
        self.ratings: Dict[str, float] = {}

    def get(self, name: str) -> float:
        return self.ratings.get(name, 1000.0)

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def update(self, a: str, b: str, score_a: float):
        # score_a: 1 win, 0 draw, -1 loss -> convert to [0,1]
        sa = 0.5 if score_a == 0 else (1.0 if score_a > 0 else 0.0)
        ra, rb = self.get(a), self.get(b)
        ea = self.expected(ra, rb)
        eb = 1.0 - ea
        ra2 = ra + self.k * (sa - ea)
        rb2 = rb + self.k * ((1.0 - sa) - eb)
        self.ratings[a] = ra2
        self.ratings[b] = rb2


# ============================================================
# 5) TRAIN LOOP
# ============================================================

def train(
    seed: int = 1,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    total_updates: int = 5000,
    rollout_steps: int = 4096,
    n_envs: int = 16,
    lr: float = 2e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    temperature_start: float = 1.2,
    temperature_end: float = 0.6,
    snapshot_interval: int = 50,
    max_pool: int = 20,
    p_vs_random: float = 0.2,
    eval_interval: int = 200,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_str)
    writer = SummaryWriter()

    model = UTTTPVNet(in_channels=7, channels=64, n_blocks=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # opponent pool (snapshots)
    opponent_pool: List[nn.Module] = []
    snap0 = copy.deepcopy(model).to(device)
    snap0.eval()
    opponent_pool.append(snap0)

    elo = Elo(k=16.0)

    for upd in range(1, total_updates + 1):
        # temperature schedule
        t = (upd - 1) / max(total_updates - 1, 1)
        temperature = temperature_start + t * (temperature_end - temperature_start)

        # Collect rollouts
        model.eval()
        transitions, roll_stats = collect_rollouts(
            model=model,
            opponent_pool=opponent_pool,
            device=device,
            rollout_steps=rollout_steps,
            n_envs=n_envs,
            temperature=temperature,
            p_vs_random=p_vs_random,
        )

        # Compute GAE
        obs, masks, actions, logp_old, returns, adv = compute_gae(
            transitions=transitions,
            model=model,
            device=device,
            gamma=gamma,
            lam=lam,
        )

        # PPO update
        model.train()
        upd_stats = ppo_update(
            model=model,
            optimizer=optimizer,
            obs=obs,
            masks=masks,
            actions=actions,
            logp_old=logp_old,
            returns=returns,
            adv=adv,
            clip_eps=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            epochs=4,
            minibatch_size=512,
            value_clip=0.2,
        )

        # Logging
        writer.add_scalar("rollout/win_rate", roll_stats["win_rate"], upd)
        writer.add_scalar("rollout/draw_rate", roll_stats["draw_rate"], upd)
        writer.add_scalar("rollout/loss_rate", roll_stats["loss_rate"], upd)
        writer.add_scalar("rollout/episodes", roll_stats["episodes"], upd)
        writer.add_scalar("train/pi_loss", upd_stats["pi_loss"], upd)
        writer.add_scalar("train/v_loss", upd_stats["v_loss"], upd)
        writer.add_scalar("train/entropy", upd_stats["entropy"], upd)
        writer.add_scalar("train/approx_kl", upd_stats["approx_kl"], upd)
        writer.add_scalar("train/clipfrac", upd_stats["clipfrac"], upd)
        writer.add_scalar("train/explained_variance", upd_stats["explained_variance"], upd)
        writer.add_scalar("train/ret_mean", upd_stats["ret_mean"], upd)
        writer.add_scalar("train/ret_std", upd_stats["ret_std"], upd)
        writer.add_scalar("train/temperature", temperature, upd)

        # Snapshot update
        if upd % snapshot_interval == 0:
            snap = copy.deepcopy(model).to(device)
            snap.eval()
            opponent_pool.append(snap)
            if len(opponent_pool) > max_pool:
                opponent_pool.pop(0)

        # Periodic evaluation
        if upd % eval_interval == 0:
            model.eval()

            # vs random baseline (random policy implemented as a "model")
            # We'll approximate random by using a snapshot that samples uniformly:
            # here we evaluate via play_match(current vs best_snapshot) + custom random matches separately.
            # Evaluate vs best snapshot (last one in pool is usually strongest recent)
            best = opponent_pool[-1]
            eval_vs_snap = play_match(model, best, device, n_games=80, temperature=0.1)

            # Evaluate vs random with a small helper model that samples uniformly from legal moves
            # We'll just do explicit random opponent here:
            rng = np.random.default_rng()
            env = UTTTEnv(micro_win_reward=0.0)
            wins = draws = losses = 0
            for _ in range(80):
                o = env.reset(seed=int(rng.integers(0, 10_000_000)))
                a_sign = 1 if rng.integers(0, 2) == 0 else -1
                while not env.done:
                    if env.current_player == a_sign:
                        x, m = obs_to_torch(o, device)
                        a, _, _ = act(model, x, m, temperature=0.1)
                    else:
                        a = act_random(o["action_mask"], rng)
                    o, _, _, _ = env.step(a)
                outcome = env.winner if env.winner is not None else 0
                score = outcome * a_sign
                if score > 0: wins += 1
                elif score < 0: losses += 1
                else: draws += 1
            eval_vs_random = {"win_rate": wins/80, "draw_rate": draws/80, "loss_rate": losses/80}

            writer.add_scalar("eval/vs_snapshot_win", eval_vs_snap["win_rate"], upd)
            writer.add_scalar("eval/vs_snapshot_draw", eval_vs_snap["draw_rate"], upd)
            writer.add_scalar("eval/vs_snapshot_loss", eval_vs_snap["loss_rate"], upd)

            writer.add_scalar("eval/vs_random_win", eval_vs_random["win_rate"], upd)
            writer.add_scalar("eval/vs_random_draw", eval_vs_random["draw_rate"], upd)
            writer.add_scalar("eval/vs_random_loss", eval_vs_random["loss_rate"], upd)

            # Update a very simple Elo between "current" and "best_snapshot"
            # Score based on win/draw/loss of current vs snapshot.
            # Convert rates to expected mean score in {-1,0,1} then update once.
            mean_score = eval_vs_snap["win_rate"] - eval_vs_snap["loss_rate"]
            elo.update("current", "snapshot_best", mean_score)
            writer.add_scalar("eval/elo_current", elo.get("current"), upd)

            print(
                f"[upd={upd:05d}] "
                f"roll win/draw/loss={roll_stats['win_rate']:.2f}/{roll_stats['draw_rate']:.2f}/{roll_stats['loss_rate']:.2f} "
                f"evalR win={eval_vs_random['win_rate']:.2f} "
                f"evalS win={eval_vs_snap['win_rate']:.2f} "
                f"KL={upd_stats['approx_kl']:.4f} ent={upd_stats['entropy']:.3f} EV={upd_stats['explained_variance']:.2f}"
            )

    writer.close()
    return model


if __name__ == "__main__":
    train(
        seed=1,
        device_str="cuda" if torch.cuda.is_available() else "cpu",
        total_updates=2000,
        rollout_steps=4096,
        n_envs=16,
        lr=2e-4,
        gamma=0.99,
        lam=0.95,
        temperature_start=1.2,
        temperature_end=0.6,
        snapshot_interval=50,
        max_pool=20,
        p_vs_random=0.2,
        eval_interval=200,
    )
