import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from enum import Enum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def idx_to_coord(a):
    return a // 3, a % 3


def coord_to_idx(i, j):
    return i * 3 + j

class Memory:
    def __init__(self, n_state: torch.Tensor, offset: int):
        self.n_state = n_state
        self.offset = offset
        pass


class Value(nn.Module):
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
        self.optim = torch.optim.Adam(self.value.parameters(), lr=lr)

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


def win(board):
    for i in range(3):
        if (board[i, 0] == 1 and board[i, 1] == 1 and board[i, 2] == 1) or (
            board[0, i] == 1 and board[1, i] == 1 and board[2, i] == 1
        ):
            return True
    if (board[0, 0] == 1 and board[1, 1] == 1 and board[2, 2] == 1) or (
        board[0, 2] == 1 and board[1, 1] == 1 and board[2, 0] == 1
    ):
        return True
    return False


class InnerGame:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.first = np.zeros(shape=[3, 3], dtype=int)
        self.second = np.zeros(shape=[3, 3], dtype=int)
        self.mask = np.ones(shape=[3, 3], dtype=int)
        self.disabled = np.zeros(shape=[3, 3], dtype=int)
        self.status = Status.PENDING

    def state(self, player_id):
        state = np.zeros(shape=[2, 3, 3], dtype=float)
        state[0] = self.first if player_id == 1 else self.second
        state[1] = self.second if player_id == 1 else self.first
        return state, self.status, self.mask

    def step(self, a, player):
        # vérifie validité
        if a < 0 or a >= 9:
            raise ValueError("Action out of range")
        r,c = idx_to_coord(a)
        
        if self.mask[r, c] != 1:
            raise Exception("Coup illégal")
        board = self.first if player == 1 else self.second
        board[r, c] = 1
        self.mask[r, c] = 0
        if win(board):
            self.status = Status.P1 if player == 1 else Status.P2
        elif self.done():
            self.status = Status.DEUCE

    def win(self, player) -> bool:
        return self.status == Status.P1 if player == 1 else self.status == Status.P2

    def done(self):
        return 0 not in (self.first + self.second)

    def valid_mask(self):
        if self.status == Status.PENDING:
            return self.mask.reshape(-1).astype(int)
        else:
            return self.disabled.reshape(-1).astype(int)


class Status(Enum):
    PENDING = 1
    DEUCE = 2
    P1 = 3
    P2 = 4


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


softmax = nn.Softmax(dim=0)


def cannonical_state(state) -> torch.Tensor:
    x = torch.Tensor(state).to(device)
    x = x[0, :, :].flatten() + x[1, :, :].flatten() * -1
    return x


def select_action_by_value(state, mask, debug=False, train=False, temp=1.0):
    best_v = -1e9
    best_a = None

    x = cannonical_state(state)
    mask_values = torch.tensor(mask, dtype=torch.bool, device=device).flatten()

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


env = InnerGame()


def mesure():
    l_wins = 0
    l_deuce = 0
    l_loss = 0

    for i in range(100):
        done = False
        player = 1
        me = True
        env.reset()
        status = Status.DEUCE
        while not done:
            s, status, mask = env.state(player if me else (3 - player))
            a = select_action_by_value(
                state=s, mask=mask, train=False
            )
            env.step(a, player if me else (3 - player))
            s, status, mask = env.state(player if me else (3 - player))
            me = not me
            done = status != Status.PENDING
        reward = 1 if status == Status.P1 else (-1 if status == Status.P2 else 0)

        if reward == 1:
            l_wins += 1
        elif reward == 0:
            l_deuce += 1
        else:
            l_loss += 1

    return (l_wins, l_deuce, l_loss)


i_win, i_deuce, i_loss = mesure()
print(f"{i_win},{i_deuce},{i_loss}")




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
    env.reset()
    memory = []
    done = False
    player = 1
    me = True
    env.reset()
    status = Status.DEUCE
    while not done:
        fun = model
        s, status, mask = env.state(player if me else (3 - player))
        a = select_action_by_value(
            state=s, mask=mask, train=True, temp=0.5 if i < 1000 else 0.1
        )
        env.step(a, player if me else (3 - player))
        s, status, mask = env.state(player if me else (3 - player))
        me = not me
        done = status != Status.PENDING
        frame = Memory(n_state=cannonical_state(s), offset=1 if me else -1)
        frames = augment_d4(frame)
        for frames_index in range(len(frames)):
            memory.append(frames[frames_index])

    # final reward to last move of player 1
    reward = 1.0 if (status == Status.P1) else (-1 if (status == Status.P2) else 0)
    first_player_losses.append(1 if reward == -1 else 0)
    first_player_deuces.append(1 if reward == 0 else 0)
    first_player_win.append(1 if reward == 1 else 0)

    states = [m.n_state for m in memory]
    z = [m.offset * reward for m in memory]
    states = torch.stack(states, dim=0)
    z = torch.Tensor(z)

    loss = model.get_losses(next_states=states, z=z)
    model.update_parameters(loss)

    learning_losses.append(loss.detach().cpu().numpy())

    # if i > rolling_length:
    #     writer.add_scalar("Wins", np.mean(first_player_win[-100:]), i)
    #     writer.add_scalar("Losses", np.mean(first_player_losses[-100:]), i)
    #     writer.add_scalar("Deuces", np.mean(first_player_deuces[-100:]), i)
    #     writer.add_scalar("mse_loss", np.mean(learning_losses[-100:]), i)


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
# plt.show(block=False)
plt.show()

i_win, i_deuce, i_loss = mesure()
print(f"{i_win},{i_deuce},{i_loss}")

player = None
