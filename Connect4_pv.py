from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
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


model = PVNet()

rolling_length = 20
learning_value_losses = []
learning_policy_losses = []
first_player_win = []
first_player_losses = []
first_player_deuces = []

change_level_episode = []

softmax = nn.Softmax(dim=0)



def select_action_by_value(
    env, agent, debug=False, train=False, temp=1.0, target=False
):
    observation, _, _, _, _ = env.last()

    x, mask = obs_to_tensor(observation, device)
    logits, v = agent(x.unsqueeze(dim=0)).squeeze(0)
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
                    env, debug=i == 0 and first_play, agent=model
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


obs_tensors = []

for env in envs:
    env.reset()
    obs, _, _, _, _ = env.last()
    obs_tensors.append(obs_to_tensor(obs["observation"], device))

for episode in range(EPISODE):

    if episode > 0 and episode % 500 == 0:
        print(episode)
    memory = [[] for i in range(len(envs))]

    temperature = 0.1 + 0.4 * (max(0, EPISODE - 2 * episode) / EPISODE)

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

            state, mask = obs_to_tensor(obs["observation"], device)

            action, _, _ = select_action_by_value(
                env, model, train=True, temp=temperature
            )

            env.step(action)

            obs2, reward2, term2, trunc2, _ = env.last()
            next_state, _ = obs_to_tensor(obs2["observation"], device)

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

    # value_loss = value.get_losses(memories=memory, gamma=GAMMA, n_step=TD_N)
    # value.update_parameters(value_loss)
    # policy_loss = policy.get_losses(
    #     memories=memory, critic=value, gamma=GAMMA, n_step=TD_N
    # )
    # policy.update_parameters(policy_loss)

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
                    env_manual, debug=True, agent=model, train=False
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
