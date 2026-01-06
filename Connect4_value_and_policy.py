import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.utils import agent_selector
import torch
import torch.nn as nn
from pettingzoo.classic import connect_four_v3
from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import time

# https://pettingzoo.farama.org/environments/classic/connect_four/
writer = SummaryWriter()
start = time.time()


class Memory:
    def __init__(self, state, mask, n_state, reward, done, action):
        self.state = state
        self.n_state = n_state
        self.reward = reward
        self.done = done
        self.action = action
        self.mask = mask


class Policy(nn.Module):
    def __init__(
        self,
        device: torch.device,
        lr: float,
        n_envs: int,
    ) -> None:
        super().__init__()
        self.device = device
        self.n_envs = n_envs

        conv_layers = [
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        fc_layers = [nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 7)]

        # define actor and critic networks
        self.conv = nn.Sequential(*conv_layers).to(self.device)
        self.head = nn.Sequential(*fc_layers).to(self.device)
        all_params = list(self.conv.parameters()) + list(self.head.parameters())
        pytorch_total_params = sum(p.numel() for p in self.conv.parameters()) + sum(
            p.numel() for p in self.head.parameters()
        )
        print(f"Number of parameters -> {pytorch_total_params}")
        self.optim = torch.optim.Adam(all_params, lr=lr)

    def mirror_logits(self, logits):
        # logits: (B, 7)
        return logits.flip(dims=[1])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state: (B, C, H, W)
        logits_list = []

        # identité
        conv = self.conv(state)
        mean = conv.mean(dim=(2, 3))
        logits = self.head(mean)
        logits_list.append(logits)

        # miroir horizontal
        state_m = torch.flip(state, dims=[2])  # W
        conv_m = self.conv(state_m)
        mean_m = conv_m.mean(dim=(2, 3))
        logits_m = self.head(mean_m)

        # reprojection action → repère canonique
        logits_m = self.mirror_logits(logits_m)
        logits_list.append(logits_m)

        # pooling équivariant
        return torch.mean(torch.stack(logits_list, dim=0), dim=0)

    def get_losses(self, memories: list[list[Memory]], critic: nn.Module, gamma, n_step=3):

        all_actor_loss = []
        all_entropy_bonus = []

        for memory in memories:
            states = torch.stack([m.state for m in memory])
            T = len(states)
            actions = torch.tensor([m.action for m in memory], device=device)
            masks = torch.stack([m.mask for m in memory])
            r = torch.tensor([m.reward for m in memory], device=device)
            done = torch.tensor([m.done for m in memory], device=device).bool()

            logits = self(states)
            logits[~masks] = -1e9
            probs = F.softmax(logits, dim=-1)

            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()
            advantages = torch.zeros(T, device=device)

            with torch.no_grad():
                values = critic(states, True).squeeze(-1)
                # n_values = -critic(new_states, True).squeeze(-1)
                for t in range(T):
                    G = 0.0
                    discount = 1.0
                    last_idx = None

                    for k in range(n_step):
                        idx = t + k
                        if idx >= T:
                            break

                        G += discount * r[idx]
                        discount *= -gamma
                        last_idx = idx

                        if done[idx]:
                            break

                    if last_idx is not None:
                        bootstrap_idx = last_idx + 1
                        if bootstrap_idx < T and not done[last_idx]:
                            G += discount * values[bootstrap_idx].item()
                    advantages[t] = G - values[t]

            actor_loss = -(log_probs * advantages.detach())
            entropy_bonus = ENTROPY * entropies
            all_actor_loss.append(actor_loss)
            all_entropy_bonus.append(entropy_bonus)
        all_actor_loss = torch.stack(all_actor_loss)
        all_entropy_bonus = torch.stack(all_entropy_bonus)
        return all_actor_loss.mean() - all_entropy_bonus.mean()

    def update_parameters(self, actor_loss) -> None:
        self.optim.zero_grad()
        loss = actor_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optim.step()


class Value(nn.Module):
    def __init__(
        self,
        device: torch.device,
        lr: float,
        n_envs: int,
    ) -> None:
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        self.optim_ctr = 0
        self.update_target = 100

        conv_layers = [
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        fc_layers = [nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1)]

        # define actor and critic networks
        self.conv = nn.Sequential(*conv_layers).to(self.device)
        self.head = nn.Sequential(*fc_layers).to(self.device)
        self.conv_target = nn.Sequential(*conv_layers).to(self.device)
        self.head_target = nn.Sequential(*fc_layers).to(self.device)
        all_params = list(self.conv.parameters()) + list(self.head.parameters())
        pytorch_total_params = sum(p.numel() for p in self.conv.parameters()) + sum(
            p.numel() for p in self.head.parameters()
        )
        print(f"Number of parameters -> {pytorch_total_params}")
        self.optim = torch.optim.Adam(all_params, lr=lr)
        self.flush_target()

    def flush_target(self):
        self.conv_target.load_state_dict(self.conv.state_dict())
        self.head_target.load_state_dict(self.head.state_dict())

    def symmetries(self, state):
        # state: (C, H, W)
        return [
            state,
            torch.flip(state, dims=[2]),  # miroir horizontal
        ]

    def forward(self, state: torch.Tensor, use_target=False) -> torch.Tensor:
        conv_model = self.conv_target if use_target else self.conv
        head_model = self.head_target if use_target else self.head
        # state: (B, C, H, W)
        values = []
        for s in self.symmetries(state):
            conv = conv_model(s)
            mean = conv.mean(dim=(2, 3))
            v = head_model(mean)
            values.append(v)

        # pooling de symétrie
        return torch.mean(torch.stack(values, dim=0), dim=0)

    def get_losses(self, memories: list[list[Memory]], gamma: float, n_step=3) -> torch.Tensor:
        all_values= []
        all_targets = []
        for memory in memories:
            T = len(memory)
            states = torch.stack([m.state for m in memory]).to(self.device)
            rewards = torch.tensor(
                [m.reward for m in memory], device=self.device, dtype=torch.float
            )
            done = torch.tensor(
                [m.done for m in memory], device=self.device, dtype=torch.bool
            )

            with torch.no_grad():
                values_target = self.forward(states, use_target=True).squeeze(-1)

            targets = torch.zeros(T, device=self.device)

            for t in range(T):
                G = 0.0
                discount = 1.0
                last_idx = None

                for k in range(n_step):
                    idx = t + k
                    if idx >= T:
                        break

                    G += discount * rewards[idx]
                    discount *= -gamma
                    last_idx = idx

                    if done[idx]:
                        break

                # bootstrap seulement si non terminal
                if last_idx is not None:
                    bootstrap_idx = last_idx + 1
                    if bootstrap_idx < T and not done[last_idx]:
                        G += discount * values_target[bootstrap_idx]

                targets[t] = G

            values = self.forward(states).squeeze(-1)
            all_values.append(values)
            all_targets.append(targets)
        all_targets = torch.stack(all_targets)
        all_values = torch.stack(all_values)
        loss = F.mse_loss(all_values, all_targets, reduction="mean")
        return loss

    def update_parameters(self, loss: torch.Tensor) -> None:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.optim_ctr += 1
        if self.optim_ctr >= self.update_target:
            print("Flush target")
            self.optim_ctr = 0
            self.flush_target()


if torch.cuda.is_available():
    print("Using cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ENVS = 8
ROLLOUT_LEN = 16

envs = [connect_four_v3.env() for _ in range(N_ENVS)]
for i, env in enumerate(envs):
    env.reset(seed=42 + i)
env_manual = connect_four_v3.env(render_mode="human")

EPISODE = 10000
LR = 1e-5  # plus stable
GAMMA = 0.99
TD_N = 10
ENTROPY = 1e-3


value = Value(
    device=device,
    lr=LR,
    n_envs=1,
)
policy = Policy(
    device=device,
    lr=LR,
    n_envs=1,
)


rolling_length = 20
learning_value_losses = []
learning_policy_losses = []
first_player_win = []
first_player_losses = []
first_player_deuces = []

change_level_episode = []

softmax = nn.Softmax(dim=0)


def cannonical_state(s):
    x = torch.Tensor(s).to(device)
    x = x.permute(2, 1, 0)
    return x


def select_action_by_value(
    env, agent, debug=False, train=False, temp=1.0, target=False
):
    observation, _, _, _, _ = env.last()
    s = observation["observation"]
    x = cannonical_state(s)

    mask = observation["action_mask"]
    mask = torch.tensor(mask, device=device).bool()
    logits = agent(x.unsqueeze(dim=0)).squeeze(0)
    logits[~mask] = -1e9
    probs = F.softmax(logits / temp, dim=-1)

    if debug:
        for prob in probs:
            print(prob)

    if train:
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
    else:
        action = torch.argmax(probs)

    return action.item(), None, None


def mesure():
    l_wins = 0
    l_deuce = 0
    l_loss = 0
    for i in range(1):
        env.reset(seed=42)
        player = "player_0"

        first_play = True
        for agent in env.agent_iter():
            _, reward, termination, truncation, _ = env.last()

            learning_model_round = agent == player
            if termination or truncation:
                action = None
                if learning_model_round:
                    if reward == 1:
                        l_wins += 1
                    elif reward == 0:
                        l_deuce += 1
                    else:
                        l_loss += 1
            else:
                action, _, _ = select_action_by_value(
                    env, debug=i == 0 and first_play, agent=policy
                )

            env.step(action)
            first_play = False
    return (l_wins, l_deuce, l_loss)


i_win, i_deuce, i_loss = mesure()
print(f"{i_win},{i_deuce},{i_loss}")


def idx_to_coord(a):
    return a // 3, a % 3


def coord_to_idx(i, j):
    return i * 3 + j


def rotate_coord_90(i, j):
    return j, 2 - i


def rotate_action(action, k):
    i, j = idx_to_coord(action)
    for _ in range(k):
        i, j = rotate_coord_90(i, j)
    return coord_to_idx(i, j)


def mirror_coord(i, j):
    return i, 2 - j


def mirror_action(action):
    i, j = idx_to_coord(action)
    i, j = mirror_coord(i, j)
    return coord_to_idx(i, j)


def rotate_grid(x, k):
    return torch.rot90(x, k, dims=(0, 1))


def mirror_grid(x):
    return torch.flip(x, dims=(1,))  # miroir horizontal


def rotate_mask(mask, k):
    return rotate_grid(mask.view(3, 3), k).flatten()


def mirror_mask(mask):
    return mirror_grid(mask.view(3, 3)).flatten()


states = []

for env in envs:
    env.reset()
    obs, _, _, _, _ = env.last()
    states.append(cannonical_state(obs["observation"]))

for episode in range(EPISODE):

    if episode > 0 and episode % 500 == 0:
        print(episode)
    memory = [[] for i in range(len(envs))]

    temperature = 0.1 + 0.4 * (max(0, EPISODE - 2 * i) / EPISODE)

    for step in range(ROLLOUT_LEN):
        for i, env in enumerate(envs):
            obs, reward, term, trunc, _ = env.last()

            if term or trunc:
                agent_selection  = env.agent_selection
                if agent_selection == "player_1":
                    reward = reward * -1
                first_player_losses.append(1 if reward == -1 else 0)
                first_player_deuces.append(1 if reward == 0 else 0)
                first_player_win.append(1 if reward == 1 else 0)
                env.reset()
                obs, _, _, _, _ = env.last()

            state = cannonical_state(obs["observation"])
            mask = torch.tensor(obs["action_mask"], device=device).bool()

            action, _, _ = select_action_by_value(
                env, policy, train=True, temp=temperature
            )

            env.step(action)

            obs2, reward2, term2, trunc2, _ = env.last()
            env.observe(env.agent_selection)
            next_state = cannonical_state(obs2["observation"])

            memory[i].append(
                Memory(
                    state=state,
                    n_state=next_state,
                    reward=-reward2,  # self-play signé inchangé
                    action=action,
                    mask=mask,
                    done=term2 or trunc2,
                )
            )

    value_loss = value.get_losses(memories=memory, gamma=GAMMA, n_step=TD_N)
    value.update_parameters(value_loss)
    policy_loss = policy.get_losses(
        memories=memory, critic=value, gamma=GAMMA, n_step=TD_N
    )
    policy.update_parameters(policy_loss)

    learning_value_losses.append(value_loss.detach().cpu().numpy())
    learning_policy_losses.append(policy_loss.detach().cpu().numpy())

    if episode > rolling_length:
        writer.add_scalar("Wins", np.mean(first_player_win[-100:]), episode)
        writer.add_scalar("Losses", np.mean(first_player_losses[-100:]), episode)
        writer.add_scalar("Deuces", np.mean(first_player_deuces[-100:]), episode)
        writer.add_scalar("value_loss", np.mean(learning_value_losses[-100:]), episode)
        writer.add_scalar("policy_loss", np.mean(learning_policy_losses[-100:]), episode)


fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 5))
fig.suptitle("Training plots for A2C in the TicTacToe environment")

# entropy
axs[0].set_title("Status")
if len(first_player_win) > 0:
    first_win_moving_average = (
        np.convolve(np.array(first_player_win), np.ones(rolling_length), mode="valid")
        / rolling_length
    )
    axs[0].plot(first_win_moving_average, label="p1w")
if len(first_player_losses) > 0:
    first_loss_moving_average = (
        np.convolve(
            np.array(first_player_losses), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[0].plot(first_loss_moving_average, label="p1l")
if len(first_player_deuces) > 0:
    first_deuce_moving_average = (
        np.convolve(
            np.array(first_player_deuces), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[0].plot(first_deuce_moving_average, label="p1d")

# for i in change_level_episode:
#     axs[0][0].vlines(i, 0, 1)
# axs[0][0].plot(deuces_moving_average)
# axs[0][0].plot(losses_moving_average)
axs[0].legend()
axs[1].set_title("Value Loss")
critic_losses_moving_average = (
    np.convolve(
        np.array(learning_value_losses).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[1].plot(critic_losses_moving_average)


axs[2].set_title("Actor Loss")
policy_losses_moving_average = (
    np.convolve(
        np.array(learning_policy_losses).flatten(),
        np.ones(rolling_length),
        mode="valid",
    )
    / rolling_length
)
axs[2].plot(policy_losses_moving_average)
plt.tight_layout()
plt.show(block=False)

i_win, i_deuce, i_loss = mesure()
print(f"{i_win},{i_deuce},{i_loss}")
end = time.time()
print(f"Elapsed {end - start}")

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
                action, _, _ = select_action_by_value(
                    env_manual, debug=True, agent=policy, train=False
                )
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

env.close()
env_manual.close()
