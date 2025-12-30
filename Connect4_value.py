import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pettingzoo.classic import connect_four_v3
from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter

# https://pettingzoo.farama.org/environments/classic/connect_four/
writer = SummaryWriter()


class Memory:
    def __init__(self, state: torch.Tensor, n_state: torch.Tensor, offset: int, reward):
        self.state = state
        self.n_state = n_state
        self.offset = offset
        self.reward = reward
        pass


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

    def forward(self, state: torch.Tensor, use_target=False) -> torch.Tensor:
        conv_model = self.conv_target if use_target else self.conv
        head_model = self.head_target if use_target else self.head
        # conv_model = self.conv
        # head_model = self.head
        conv = conv_model(state)
        mean = conv.mean(dim=(2, 3))
        fc = head_model(mean)
        return fc

    def get_losses(
        self, memory: list[Memory], gamma: float
    ) -> torch.Tensor:
        T = len(memory)
        vs = []
        targets = np.zeros(T)
        for i in reversed(range(T)):
            frame = memory[i]
            s = frame.state
            v = self.forward(s.unsqueeze(dim=0)).squeeze(-1)
            vs.append(v)
            target = frame.reward
            if i < T - 1:
                ns = frame.n_state
                target += -self.forward(ns.unsqueeze(dim=0), use_target= True).item() * gamma
            targets[i] = target 

        vs = torch.cat(vs)
        targets = torch.tensor(targets, dtype=torch.float)
        loss = F.mse_loss(vs, targets, reduction="mean")
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
env = connect_four_v3.env()  # render_mode="human")
env_manual = connect_four_v3.env(render_mode="human")

EPISODE = 2000
LR = 1e-4  # plus stable
GAMMA = 0.99

model = Value(
    device=device,
    lr=LR,
    n_envs=1,
)


rolling_length = 20
learning_losses = []
first_player_win = []
first_player_losses = []
first_player_deuces = []

change_level_episode = []

softmax = nn.Softmax(dim=0)


def cannonical_state(s):
    x = torch.Tensor(s).to(device)
    x = x.permute(2, 1, 0)
    return x


def select_action_by_value(env, debug=False, train=False, temp=1.0, target= False):
    best_v = -1e9
    best_a = None
    observation, _, _, _, _ = env.last()
    s = observation["observation"]
    x = cannonical_state(s)

    mask = observation["action_mask"]
    mask_values = torch.tensor(mask, dtype=torch.bool, device=device)

    values = []
    actions = []
    for a in torch.where(mask_values)[0]:
        state = x.detach().clone()

        slots = torch.where(state[0][a] + state[1][a] == 0)[0]
        column = slots[slots.size()[0] - 1]
        state[0, a, column] = 1
        next_state = torch.stack([state[1], state[0]], dim=0)
        with torch.no_grad():
            v = -model(next_state.unsqueeze(dim=0), target)

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
    # s0 = memory.state
    # s1 = memory.n_state
    # offset = memory.offset
    memories = []
    #
    # # vue canonique
    # s1 = s1.view(3, 3)
    #
    # for k in range(4):
    #     # --- rotation ---
    #     ns_rot = rotate_grid(s1, k)
    #
    #     memories.append(Memory(n_state=ns_rot.flatten().to(device), offset=offset))
    #
    #     ns_m = mirror_grid(ns_rot)
    #
    #     memories.append(Memory(n_state=ns_m.flatten().to(device), offset=offset))
    #
    memories.append(memory)
    return memories

player = None

for i in range(EPISODE):
    if i > 0 and i % 50 == 0:
        print(i)

    env.reset(seed=42)

    memory = []
    player = "player_0" if player is None or player == "player_1" else "player_1"
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
            if agent == player:
                # game_p1_reward = 0.1 if reward == 0 else reward
                first_player_losses.append(1 if reward == -1 else 0)
                first_player_deuces.append(1 if reward == 0 else 0)
                first_player_win.append(1 if reward == 1 else 0)

            env.step(action)
        else:
            state = observation["observation"]
            state = cannonical_state(state)
            action = select_action_by_value(
                env, train=True, temp=0.5 if i < 1000 else 0.1, target = agent != player
            )
            env.step(action)
            observation, reward, termination, truncation, info = env.last()
            nstate = observation["observation"]
            x = cannonical_state(nstate)

            frame = Memory(
                state=state, n_state=x, offset=1 if agent == player else -1, reward=-reward
            )
            memory.append(frame)
            # frames = augment_d4(frame)
            # for frames_index in range(len(frames)):
            #     memory.append(frames[frames_index])

    states = [m.n_state for m in memory]
    states = torch.stack(states, dim=0)

    loss = model.get_losses(memory=memory, gamma=GAMMA)
    model.update_parameters(loss)

    learning_losses.append(loss.detach().cpu().numpy())

    if i > rolling_length:
        writer.add_scalar("Wins", np.mean(first_player_win[-100:]), i)
        writer.add_scalar("Losses", np.mean(first_player_losses[-100:]), i)
        writer.add_scalar("Deuces", np.mean(first_player_deuces[-100:]), i)
        writer.add_scalar("mse_loss", np.mean(learning_losses[-100:]), i)


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
    player = "player_0" if player is None or player == "player_1" else "player_1"
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
                ok = False
                while not ok:
                    ok = True
                    try:
                        action = input()
                        action = np.array(action, dtype=np.int16)
                    except:
                        ok = False
                        

        env_manual.step(action)

env.close()
env_manual.close()
