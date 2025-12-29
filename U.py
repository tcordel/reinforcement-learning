import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from enum import Enum
from torch.utils.tensorboard.writer import SummaryWriter

writer = SummaryWriter()

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
        pytorch_total_params = sum(p.numel() for p in self.conv.parameters()) + sum(p.numel() for p in self.head.parameters())
        print(f"Number of parameters -> {pytorch_total_params}")
        self.optim = torch.optim.Adam(all_params, lr=lr)

    def forward(self, state: torch.Tensor) -> float:
        conv = self.conv(state)
        mean = conv.mean(dim=(2, 3))
        fc = self.head(mean)
        return fc

    def get_losses(
        self,
        next_states: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        values = self.forward(next_states)
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


def select_action_by_value(agent, state, mask, debug=False, train=False, temp=1.0):
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
            v = agent(state.unsqueeze(0))

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


def mesure(a1, a2):
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
            agent = a1 if me else a2
            a = select_action_by_value(agent=agent, state=s, mask=mask, train=False)
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

a1 = Value(
    device=device,
    lr=0,
    n_envs=1,
)

a2 = Value(
    device=device,
    lr=0,
    n_envs=1,
)

a1.conv.load_state_dict(torch.load("./conv-500.pth", map_location="cpu"))
a1.head.load_state_dict(torch.load("./head-500.pth", map_location="cpu"))

a2.conv.load_state_dict(torch.load("./conv-1028500.pth", map_location="cpu"))
a2.head.load_state_dict(torch.load("./head-1028500.pth", map_location="cpu"))
i_win, i_deuce, i_loss = mesure(a1, a2)
print(f"{i_win},{i_deuce},{i_loss}")


i_win, i_deuce, i_loss = mesure(a2, a1)
print(f"{i_win},{i_deuce},{i_loss}")
