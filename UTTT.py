import numpy as np
import torch
import torch.nn as nn
from enum import Enum
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn import functional as F

writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def idx_to_coord(a):
    return a // 3, a % 3


def coord_to_idx(i, j):
    return i * 3 + j


class Node:
    def __init__(self, state, player):
        self.state = state
        self.player = player
        self.N = 0
        self.W = 0.0
        self.children = {}  # action -> Node


def mcts_search(root_state, env, value_net, n_sim=50, c_puct=1.0, gamma=0.99):
    root = Node(root_state, player=1)

    for _ in range(n_sim):
        node = root
        sim_env = env.clone()
        path = []

        # 1️⃣ Selection
        while node.children:
            best_score = -1e9
            best_action = None
            best_child = None

            for a, child in node.children.items():
                Q = child.W / child.N if child.N > 0 else 0
                U = c_puct * (node.N**0.5) / (1 + child.N)
                score = Q + U
                if score > best_score:
                    best_score = score
                    best_action = a
                    best_child = child

            sim_env.step(best_action)
            path.append(node)
            node = best_child

        # 2️⃣ Expansion
        obs, reward, done, _ = sim_env.last()
        if not done:
            legal_actions = sim_env.legal_actions()
            for a in legal_actions:
                next_state = sim_env.peek(a)
                node.children[a] = Node(next_state, -node.player)

        # 3️⃣ Evaluation (value network)
        if done:
            value = reward
        else:
            x = encode_canonical(node.state, node.player)
            value = value_net(x).item()

        # 4️⃣ Backprop
        for n in reversed(path):
            n.N += 1
            n.W += value
            value = gamma * (-value)

        node.N += 1
        node.W += value

    # 5️⃣ Action selection
    best_action = max(root.children.items(), key=lambda kv: kv[1].N)[0]

    return best_action


class Memory:
    def __init__(self, state: torch.Tensor, n_state: torch.Tensor, offset: int):
        self.state = state
        self.n_state = n_state
        self.offset = offset
        pass


class Status(Enum):
    PENDING = 1
    DEUCE = 2
    P1 = 3
    P2 = 4


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

        conv_layers = [
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        fc_layers = [nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh()]

        # define actor and critic networks
        self.conv = nn.Sequential(*conv_layers).to(self.device)
        self.head = nn.Sequential(*fc_layers).to(self.device)
        all_params = list(self.conv.parameters()) + list(self.head.parameters())
        pytorch_total_params = sum(p.numel() for p in self.conv.parameters()) + sum(
            p.numel() for p in self.head.parameters()
        )
        print(f"Number of parameters -> {pytorch_total_params}")
        self.optim = torch.optim.Adam(all_params, lr=lr)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        conv = self.conv(state)
        mean = conv.mean(dim=(2, 3))
        fc = self.head(mean)
        return fc

    def get_losses(
        self,
        memory: list[list[Memory]],
        reward: float,
        gamma: float
    ) -> torch.Tensor:
        T = len(memory)
        vs = []
        targets=[]
        for i in reversed(range(T)):
            frames = memory[i]
            s = torch.stack([f.state for f in frames])
            v = self.forward(s).squeeze(-1)
            vs.append(v)
            ns = torch.stack([f.n_state for f in frames])
            if i == T-1:
                target = torch.ones(len(frames)) * reward
            else:
                target = self.forward(ns).squeeze(-1) * gamma
            target = target * frames[0].offset
            targets.append(target)

        vs = torch.cat(vs)
        targets = torch.cat(targets)
        loss = F.mse_loss(vs, targets, reduction="mean")
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
        return state

    def step(self, a, player):
        # vérifie validité
        if a < 0 or a >= 9:
            raise ValueError("Action out of range")
        r, c = idx_to_coord(a)

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

    def to_string(self):
        print(self.first - self.second)


class Game:
    def __init__(self):
        self.games = np.array(
            [[InnerGame() for _ in range(3)] for _ in range(3)], dtype=InnerGame
        )

    def reset_action_mask(self):
        self.action_mask = np.ones(81)

    def reset(self):
        self.reset_action_mask()
        for row in self.games:
            for col in row:
                col.reset()

    def state(self, player_id):
        games = np.array(
            [[(self.games[j][i].state(player_id)) for i in range(3)] for j in range(3)],
            dtype=np.int_,
        )
        bitboard_states = np.zeros([2, 9, 9])

        row0 = [games[0][0], games[0][1], games[0][2]]
        row1 = [games[1][0], games[1][1], games[1][2]]
        row2 = [games[2][0], games[2][1], games[2][2]]
        row0 = np.concatenate(row0, axis=2)
        row1 = np.concatenate(row1, axis=2)
        row2 = np.concatenate(row2, axis=2)
        map = [row0, row1, row2]
        bitboard_states = np.concatenate(map, axis=1)
        return bitboard_states.copy(), self.status(), self.valid_mask()

    def action_to_coord(self, a) -> tuple[int, int]:
        mini_game = int(a / 9)
        mini_game_a = a - mini_game * 9

        mini_game_r = int(mini_game / 3)
        mini_game_c = int(mini_game % 3)
        next_minigame_r, next_minigame_c = idx_to_coord(mini_game_a)
        return next_minigame_r + 3 * mini_game_r, next_minigame_c + 3 * mini_game_c

    def step(self, a, player):
        mini_game = int(a / 9)
        mini_game_r = int(mini_game / 3)
        mini_game_c = int(mini_game % 3)
        mini_game_a = a - mini_game * 9
        if not self.action_mask[a] == 1:
            raise Exception("Coup illégal")

        self.games[mini_game_r][mini_game_c].step(mini_game_a, player)
        next_minigame_r, next_minigame_c = idx_to_coord(mini_game_a)
        self.reset_action_mask()
        if 1 in self.games[next_minigame_r, next_minigame_c].valid_mask():
            self.action_mask = np.zeros(81)
            self.action_mask[
                (next_minigame_r * 3 + next_minigame_c) * 9 : (
                    next_minigame_r * 3 + next_minigame_c + 1
                )
                * 9
            ] = 1

    def valid_mask(self):
        return (
            np.concat(
                [
                    self.games[0][0].valid_mask(),
                    self.games[0][1].valid_mask(),
                    self.games[0][2].valid_mask(),
                    self.games[1][0].valid_mask(),
                    self.games[1][1].valid_mask(),
                    self.games[1][2].valid_mask(),
                    self.games[2][0].valid_mask(),
                    self.games[2][1].valid_mask(),
                    self.games[2][2].valid_mask(),
                ]
            )
            .reshape(-1)
            .astype(int)
            * self.action_mask
        ).copy()

    def status(self) -> Status:
        first = np.zeros(shape=[3, 3], dtype=int)
        second = np.zeros(shape=[3, 3], dtype=int)
        for row in range(3):
            for col in range(3):
                first[row][col] = 1 if self.games[row][col].win(1) else 0
                second[row][col] = 1 if self.games[row][col].win(2) else 0
        if win(first):
            return Status.P1
        if win(second):
            return Status.P2

        if not self.valid_mask().__contains__(1):
            if first.sum() > second.sum():
                return Status.P1
            elif first.sum() < second.sum():
                return Status.P2
            else:
                return Status.DEUCE

        return Status.PENDING

    def to_string(self):
        state, _, _ = self.state(1)
        board = state[0] + state[1] * 2
        for row in range(9):
            for col in range(9):
                print(board[row, col], end="")
                if col == 2 or col == 5:
                    print("|", end="")
            print()
            if row == 2 or row == 5:
                print("-----------")
        # print(self.action_mask)

        # for row in range(3):
        #     for col in range(3):
        #         print(self.games[row, col].status, end="")
        #     print()


EPISODE = 1000
LR = 3e-4  # plus stable
GAMMA = 0.99

model = Value(
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
    # x = x[0, :, :].flatten() + x[1, :, :].flatten() * -1
    return x


def temperature(episode):
    if episode < 20_000:
        return 1.0
    elif episode < 60_000:
        return 0.5
    else:
        return 0.25


def select_action_by_value(state, mask, debug=False, train=False, temp=1.0):
    best_v = -1e9
    best_a = None

    x = cannonical_state(state)
    mask_values = torch.tensor(mask, dtype=torch.bool, device=device).flatten()

    values = []
    actions = []
    for a in torch.where(mask_values)[0]:
        state = x.detach().clone()
        r, c = env.action_to_coord(a)
        state[0, r, c] = 1

        with torch.no_grad():
            v = model(state.unsqueeze(0))

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


env = Game()


def mesure():
    l_wins = 0
    l_deuce = 0
    l_loss = 0

    for i in range(10):
        done = False
        player = 1
        me = True
        env.reset()
        status = Status.DEUCE
        while not done:
            s, status, mask = env.state(player if me else (3 - player))
            a = select_action_by_value(state=s, mask=mask, train=False)
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


WIDTH = 9


def rotate_coord_90(i, j):
    return j, (WIDTH - 1) - i


def mirror_coord(i, j):
    return i, (WIDTH - 1) - j


def rotate_grid(x, k):
    return torch.rot90(x, k, dims=(1, 2))


def mirror_grid(x):
    return torch.flip(x, dims=(1,))  # miroir horizontal


def rotate_mask(mask, k):
    return rotate_grid(mask.view(WIDTH, WIDTH), k).flatten()


def mirror_mask(mask):
    return mirror_grid(mask.view(WIDTH, WIDTH)).flatten()


def augment_d4(memory: Memory) -> list[Memory]:
    """
    Retourne une liste de Memory
    """
    s0 = memory.state
    s1 = memory.n_state
    offset = memory.offset
    memories = []

    for k in range(4):
        # --- rotation ---
        s_rot = rotate_grid(s0, k)
        ns_rot = rotate_grid(s1, k)

        memories.append(Memory(state=s_rot, n_state=ns_rot.to(device), offset=offset))

        s_m = mirror_grid(s_rot)
        ns_m = mirror_grid(ns_rot)

        memories.append(Memory(state=s_m, n_state=ns_m.to(device), offset=offset))

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
        temp = temperature(i)
        a = select_action_by_value(state=s, mask=mask, train=True, temp=temp)
        env.step(a, player if me else (3 - player))
        ns, status, mask = env.state(player if me else (3 - player))
        me = not me
        done = status != Status.PENDING
        frame = Memory(
            state=cannonical_state(s),
            n_state=cannonical_state(ns),
            offset=1 if me else -1,
        )
        frames = augment_d4(frame)
        for frames_index in range(len(frames)):
            memory.append(frames)

    # final reward to last move of player 1
    reward = 1.0 if (status == Status.P1) else (-1 if (status == Status.P2) else 0)
    first_player_losses.append(1 if reward == -1 else 0)
    first_player_deuces.append(1 if reward == 0 else 0)
    first_player_win.append(1 if reward == 1 else 0)

    loss = model.get_losses(memory=memory, reward=reward, gamma=GAMMA)
    model.update_parameters(loss)

    learning_losses.append(loss.detach().cpu().numpy())

    if i > 0 and i % 500 == 0:
        torch.save(model.conv.state_dict(), f"./conv-{i}.pth")
        torch.save(model.head.state_dict(), f"./head-{i}.pth")

    if i > rolling_length:
        writer.add_scalar("Wins", np.mean(first_player_win[-100:]), i)
        writer.add_scalar("Losses", np.mean(first_player_losses[-100:]), i)
        writer.add_scalar("Deuces", np.mean(first_player_deuces[-100:]), i)
        writer.add_scalar("mse_loss", np.mean(learning_losses[-100:]), i)


# fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))
# fig.suptitle("Training plots for A2C in the TicTacToe environment")
#
# # entropy
# axs[0].set_title("Status")
# if len(first_player_win) > 0:
#     first_win_moving_average = (
#         np.convolve(np.array(first_player_win), np.ones(rolling_length), mode="valid")
#         / rolling_length
#     )
#     axs[0].plot(first_win_moving_average, label="p1w")
# if len(first_player_losses) > 0:
#     first_loss_moving_average = (
#         np.convolve(
#             np.array(first_player_losses), np.ones(rolling_length), mode="valid"
#         )
#         / rolling_length
#     )
#     axs[0].plot(first_loss_moving_average, label="p1l")
# if len(first_player_deuces) > 0:
#     first_deuce_moving_average = (
#         np.convolve(
#             np.array(first_player_deuces), np.ones(rolling_length), mode="valid"
#         )
#         / rolling_length
#     )
#     axs[0].plot(first_deuce_moving_average, label="p1d")
#
# # for i in change_level_episode:
# #     axs[0][0].vlines(i, 0, 1)
# # axs[0][0].plot(deuces_moving_average)
# # axs[0][0].plot(losses_moving_average)
# axs[0].set_xlabel("Number of updates")
# axs[0].legend()
# #  loss
# axs[1].set_title("Loss")
# critic_losses_moving_average = (
#     np.convolve(
#         np.array(learning_losses).flatten(), np.ones(rolling_length), mode="valid"
#     )
#     / rolling_length
# )
# axs[1].plot(critic_losses_moving_average)
# axs[1].set_xlabel("Number of updates")
#
#
# plt.tight_layout()
# # plt.show(block=False)
# plt.show()

i_win, i_deuce, i_loss = mesure()
print(f"{i_win},{i_deuce},{i_loss}")
