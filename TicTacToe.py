import matplotlib.pyplot as plt
import numpy as np
from numpy._core.numeric import roll
import torch
import torch.nn as nn
import random
from pettingzoo.classic import tictactoe_v3
from torch import cpu, optim
from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter

writer = SummaryWriter()

class Memory:
    def __init__(self, n_state: torch.Tensor, offset: int):
        self.n_state = n_state
        self.offset = offset
        pass


class Value(nn.Module):
    """
    (Synchronous) Advantage Actor-Critic agent class

    Args:
        n_features: The number of features of the input state.
        n_actions: The number of actions the agent can take.
        device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
                for this code CPU is totally fine).
        critic_lr: The learning rate for the critic network (should usually be larger than the actor_lr).
        actor_lr: The learning rate for the actor network.
        n_envs: The number of environments that run in parallel (on multiple CPUs) to collect experiences.
    """

    def __init__(
        self,
        n_features: int,
        device: torch.device,
        lr: float,
        n_envs: int,
    ) -> None:
        super().__init__()
        self.device = device
        self.n_envs = n_envs

        actor_layers = [
            nn.Linear(n_features, 32),
            # nn.ReLU(),
            # nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(
                32, 1
            ),  # estimate action logits (will be fed into a softmax later)
            nn.Tanh(),
        ]

        # define actor and critic networks
        self.value = nn.Sequential(*actor_layers).to(self.device)
        pytorch_total_params = sum(p.numel() for p in self.value.parameters())
        print(f"Number of parameters -> {pytorch_total_params}")
        self.optim = optim.Adam(self.value.parameters(), lr=lr)

    def forward(self, state: torch.Tensor) -> float:
        return self.value(state).squeeze(-1)

    def get_losses(
        self,
        next_states: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        values = self.value(next_states)
        loss = F.mse_loss(values.squeeze(-1), z)
        return loss

    def update_parameters(self, loss: torch.Tensor) -> None:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


if torch.cuda.is_available():
    print("Using cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = tictactoe_v3.env()  # render_mode="human")
env_manual = tictactoe_v3.env(render_mode="human")

EPISODE = 5000
LR = 1e-2  # plus stable

model = Value(
    n_features=9,
    device=device,
    lr=LR,
    n_envs=1,
)


rolling_length = 100
learning_losses = []
first_player_win = []
first_player_losses = []
first_player_deuces = []

change_level_episode = []

softmax = nn.Softmax(dim=0)


def select_action_by_value(env, debug=False, train=False, temp=1.0):
    best_v = -1e9
    best_a = None
    observation, _, _, _, _ = env.last()
    s = observation["observation"]
    x = torch.Tensor(s).to(device)
    x = x[:, :, 0].flatten() + x[:, :, 1].flatten() * -1

    mask = observation["action_mask"]
    mask_values = torch.tensor(mask, dtype=torch.bool, device=device)

    values = []
    actions = []
    for a in torch.where(mask_values)[0]:
        state = x.detach().clone()
        state[a] = 1

        with torch.no_grad():
            v = model(state)

        values.append(v.item())
        actions.append(a.item())
        if debug:
            print(f"{a} -> {v}")
        if v > best_v:
            best_v = v
            best_a = a.item()

    if train:
        probs = softmax(torch.Tensor(values).to(device) / temp).detach().cpu().numpy()
        return np.random.choice(actions, p=probs)
    else:
        return best_a


def mesure():
    l_wins = 0
    l_deuce = 0
    l_loss = 0
    for i in range(100):
        env.reset(seed=42)
        player = "player_1"

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
                action = select_action_by_value(env, i == 0 and first_play)

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


def augment_d4(memory: Memory) -> list[Memory]:
    """
    Retourne une liste de Memory
    """
    s1 = memory.n_state
    offset = memory.offset
    memories = []

    # vue canonique
    s1 = s1.view(3, 3)

    for k in range(4):
        # --- rotation ---
        ns_rot = rotate_grid(s1, k)

        memories.append(Memory(n_state=ns_rot.flatten().to(device), offset=offset))

        ns_m = mirror_grid(ns_rot)

        memories.append(Memory(n_state=ns_m.flatten().to(device), offset=offset))

    # memories.append(memory)
    return memories


for i in range(EPISODE):
    env.reset(seed=42)

    memory = []
    game_p1_reward = 0
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
            if agent == "player_1":
                # game_p1_reward = 0.1 if reward == 0 else reward
                game_p1_reward = reward
                first_player_losses.append(1 if reward == -1 else 0)
                first_player_deuces.append(1 if reward == 0 else 0)
                first_player_win.append(1 if reward == 1 else 0)

            env.step(action)
        else:
            action = select_action_by_value(
                env, train=True, temp=0.5 if i < 1000 else 0.1
            )
            env.step(action)
            observation, reward, termination, truncation, info = env.last()
            state = observation["observation"]
            mask = observation["action_mask"]
            mask_values = torch.tensor(mask, dtype=torch.bool, device=device)
            x = torch.Tensor(state).to(device)
            x = x[:, :, 1].flatten() + x[:, :, 0].flatten() * -1

            frame = Memory(n_state=x, offset=1 if agent == "player_1" else -1)
            frames = augment_d4(frame)
            for frames_index in range(len(frames)):
                memory.append(frames[frames_index])

    states = [m.n_state for m in memory]
    z = [m.offset * game_p1_reward for m in memory]
    states = torch.stack(states, dim=0)
    z = torch.Tensor(z)

    loss = model.get_losses(next_states=states, z=z)
    model.update_parameters(loss)

    learning_losses.append(loss.detach().cpu().numpy())

    if i > rolling_length:
        writer.add_scalar('Wins', np.mean(first_player_win[-100:]), i)
        writer.add_scalar('Losses', np.mean(first_player_losses[-100:]), i)
        writer.add_scalar('Deuces', np.mean(first_player_deuces[-100:]), i)
        writer.add_scalar('mse_loss', np.mean(learning_losses[-100:]), i)


fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))
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
axs[0].set_xlabel("Number of updates")
axs[0].legend()
#  loss
axs[1].set_title("Loss")
critic_losses_moving_average = (
    np.convolve(
        np.array(learning_losses).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[1].plot(critic_losses_moving_average)
axs[1].set_xlabel("Number of updates")


plt.tight_layout()
plt.show(block=False)

i_win, i_deuce, i_loss = mesure()
print(f"{i_win},{i_deuce},{i_loss}")

player = None
while True:
    env_manual.reset(seed=42)
    player = "player_2" if player is None or player == "player_1" else "player_1"
    for agent in env_manual.agent_iter():
        observation, reward, termination, truncation, info = env_manual.last()

        learning_model_round = agent == player
        if termination or truncation:
            action = None
        else:
            if learning_model_round:
                action = select_action_by_value(env_manual)
            else:
                print("Pick action")
                action = input()
                action = np.array(action, dtype=np.int16)

        env_manual.step(action)

env.close()
env_manual.close()
