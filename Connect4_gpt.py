# az_connect4.py
# AlphaZero-like for Connect4 (CPU-friendly), PyTorch
import math
import random
from dataclasses import dataclass
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Connect4 environment (fast)
# ----------------------------

H, W = 6, 7

def check_winner(board: np.ndarray) -> int:
    """
    board: (6,7) with {-1,0,+1}
    returns: +1 if player +1 wins, -1 if player -1 wins, 0 otherwise
    """
    b = board
    # horizontal
    for r in range(H):
        for c in range(W - 3):
            s = b[r, c:c+4].sum()
            if s == 4: return 1
            if s == -4: return -1
    # vertical
    for r in range(H - 3):
        for c in range(W):
            s = b[r:r+4, c].sum()
            if s == 4: return 1
            if s == -4: return -1
    # diag down-right
    for r in range(H - 3):
        for c in range(W - 3):
            s = b[r, c] + b[r+1, c+1] + b[r+2, c+2] + b[r+3, c+3]
            if s == 4: return 1
            if s == -4: return -1
    # diag up-right
    for r in range(3, H):
        for c in range(W - 3):
            s = b[r, c] + b[r-1, c+1] + b[r-2, c+2] + b[r-3, c+3]
            if s == 4: return 1
            if s == -4: return -1
    return 0

def legal_moves(board: np.ndarray) -> np.ndarray:
    # legal if top cell empty
    return (board[0, :] == 0)

def apply_move(board: np.ndarray, col: int, player: int) -> np.ndarray:
    """
    returns new board after dropping player stone into col
    """
    nb = board.copy()
    for r in range(H-1, -1, -1):
        if nb[r, col] == 0:
            nb[r, col] = player
            return nb
    raise ValueError("Illegal move (full column)")

def is_full(board: np.ndarray) -> bool:
    return np.all(board[0, :] != 0)

@dataclass(frozen=True)
class State:
    board: bytes  # board.tobytes() for hashing
    player: int   # player-to-move: +1 or -1

def encode(board: np.ndarray, player: int, device: torch.device) -> torch.Tensor:
    """
    Encode from the perspective of 'player' (player-to-move):
    plane0 = current player's stones
    plane1 = opponent's stones
    returns tensor (2,6,7) float32
    """
    cur = (board == player).astype(np.float32)
    opp = (board == -player).astype(np.float32)
    x = np.stack([cur, opp], axis=0)  # (2,6,7)
    return torch.from_numpy(x).to(device)

def mirror_board(board: np.ndarray) -> np.ndarray:
    return np.flip(board, axis=1).copy()

def mirror_pi(pi: np.ndarray) -> np.ndarray:
    return pi[::-1].copy()

# ----------------------------
# Network (ResNet policy+value)
# ----------------------------

class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)

class AZNet(nn.Module):
    def __init__(self, ch=64, blocks=4):
        super().__init__()
        self.c0 = nn.Conv2d(2, ch, 3, padding=1, bias=False)
        self.b0 = nn.BatchNorm2d(ch)
        self.res = nn.Sequential(*[ResidualBlock(ch) for _ in range(blocks)])

        # policy head
        self.pc = nn.Conv2d(ch, 2, 1, bias=False)
        self.pb = nn.BatchNorm2d(2)
        self.pl = nn.Linear(2 * H * W, W)

        # value head
        self.vc = nn.Conv2d(ch, 1, 1, bias=False)
        self.vb = nn.BatchNorm2d(1)
        self.v1 = nn.Linear(H * W, 64)
        self.v2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B,2,6,7)
        x = F.relu(self.b0(self.c0(x)))
        x = self.res(x)

        # policy
        p = F.relu(self.pb(self.pc(x)))
        p = p.flatten(1)
        logits = self.pl(p)  # (B,7)

        # value
        v = F.relu(self.vb(self.vc(x)))
        v = v.flatten(1)
        v = F.relu(self.v1(v))
        v = torch.tanh(self.v2(v)).squeeze(-1)  # (B,)
        return logits, v

# ----------------------------
# MCTS (PUCT)
# ----------------------------

class Node:
    __slots__ = ("board", "player", "P", "N", "W", "children", "expanded", "terminal", "winner")
    def __init__(self, board: np.ndarray, player: int):
        self.board = board
        self.player = player
        self.P = np.zeros(W, dtype=np.float32)
        self.N = np.zeros(W, dtype=np.int32)
        self.W = np.zeros(W, dtype=np.float32)
        self.children = {}  # action -> Node
        self.expanded = False
        self.winner = check_winner(board)
        self.terminal = (self.winner != 0) or is_full(board)

    def q(self):
        q = np.zeros(W, dtype=np.float32)
        nz = self.N > 0
        q[nz] = self.W[nz] / self.N[nz]
        return q

    def select_action(self, c_puct: float) -> int:
        legal = legal_moves(self.board)
        q = self.q()
        # PUCT: Q + U
        sumN = self.N.sum()
        sqrt_sum = math.sqrt(sumN + 1e-8)
        u = c_puct * self.P * (sqrt_sum / (1.0 + self.N))
        scores = q + u
        scores[~legal] = -1e9
        return int(np.argmax(scores))

    def next_node(self, action: int) -> "Node":
        if action in self.children:
            return self.children[action]
        nb = apply_move(self.board, action, self.player)
        child = Node(nb, -self.player)
        self.children[action] = child
        return child

def softmax_masked(logits: torch.Tensor, legal_mask: np.ndarray) -> np.ndarray:
    # logits: (7,) torch
    mask = torch.tensor(legal_mask, device=logits.device, dtype=torch.bool)
    x = logits.clone()
    x[~mask] = -1e9
    p = torch.softmax(x, dim=-1).detach().cpu().numpy().astype(np.float32)
    # numerical safety
    p = p * legal_mask.astype(np.float32)
    s = p.sum()
    if s <= 0:
        # fallback uniform over legal
        p = legal_mask.astype(np.float32)
        p /= p.sum()
    else:
        p /= s
    return p

@torch.no_grad()
def expand_with_net(node: Node, net: AZNet, device: torch.device, add_dirichlet: bool,
                    dir_alpha=0.3, dir_eps=0.25):
    if node.expanded or node.terminal:
        return
    x = encode(node.board, node.player, device).unsqueeze(0)  # (1,2,6,7)
    logits, v = net(x)
    legal = legal_moves(node.board)
    p = softmax_masked(logits.squeeze(0), legal)
    if add_dirichlet:
        legal_idx = np.where(legal)[0]
        noise = np.random.dirichlet([dir_alpha] * len(legal_idx)).astype(np.float32)
        p2 = p.copy()
        p2[legal_idx] = (1 - dir_eps) * p2[legal_idx] + dir_eps * noise
        p = p2
        p /= p.sum()
    node.P = p
    node.expanded = True
    return float(v.item())

def terminal_value_from_perspective(node: Node) -> float:
    # winner is in absolute players; node.player is player-to-move
    # value should be from node.player perspective
    if node.winner == 0:
        return 0.0
    return float(node.winner * node.player)  # +1 if current player is winner, else -1

def mcts_search(root: Node, net: AZNet, device: torch.device,
                sims: int = 200, c_puct: float = 1.5,
                add_dirichlet: bool = True) -> np.ndarray:
    # ensure root prior exists
    if not root.expanded and not root.terminal:
        expand_with_net(root, net, device, add_dirichlet=add_dirichlet)

    for _ in range(sims):
        node = root
        path = []  # (node, action)
        # selection
        while node.expanded and not node.terminal:
            a = node.select_action(c_puct)
            path.append((node, a))
            node = node.next_node(a)

        # evaluation / expansion
        if node.terminal:
            v = terminal_value_from_perspective(node)
        else:
            v = expand_with_net(node, net, device, add_dirichlet=False)

        # backprop (flip perspective each ply)
        for n, a in reversed(path):
            n.N[a] += 1
            n.W[a] += v
            v = -v

    counts = root.N.astype(np.float32)
    if counts.sum() <= 0:
        # fallback
        legal = legal_moves(root.board).astype(np.float32)
        legal /= legal.sum()
        return legal
    pi = counts / counts.sum()
    return pi

def select_action_from_pi(pi: np.ndarray, temperature: float) -> int:
    if temperature <= 1e-6:
        return int(np.argmax(pi))
    p = pi ** (1.0 / temperature)
    p /= p.sum()
    return int(np.random.choice(np.arange(W), p=p))

# ----------------------------
# Replay buffer + training
# ----------------------------

@dataclass
class Sample:
    x: np.ndarray   # (2,6,7) float32 on CPU
    pi: np.ndarray  # (7,) float32
    z: float        # [-1,0,1]

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def add_game(self, samples: list[Sample], augment_mirror: bool = True):
        for s in samples:
            self.buf.append(s)
            if augment_mirror:
                # mirror x and pi
                x = s.x.reshape(2, H, W)
                xm = np.flip(x, axis=2).copy()
                pim = s.pi[::-1].copy()
                self.buf.append(Sample(x=xm, pi=pim, z=s.z))

    def sample_batch(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buf, batch_size)
        x = torch.tensor(np.stack([b.x for b in batch], axis=0), device=device)
        pi = torch.tensor(np.stack([b.pi for b in batch], axis=0), device=device)
        z = torch.tensor([b.z for b in batch], device=device, dtype=torch.float32)
        return x, pi, z

    def __len__(self):
        return len(self.buf)

def loss_fn(logits: torch.Tensor, v: torch.Tensor, pi_target: torch.Tensor, z_target: torch.Tensor):
    # policy loss: cross-entropy with soft targets
    logp = F.log_softmax(logits, dim=-1)
    policy_loss = -(pi_target * logp).sum(dim=1).mean()
    value_loss = F.mse_loss(v, z_target)
    return policy_loss, value_loss, policy_loss + value_loss

@torch.no_grad()
def play_self_game(net: AZNet, device: torch.device,
                   sims: int, c_puct: float,
                   temp_moves: int = 8, temp: float = 1.0) -> list[Sample]:
    board = np.zeros((H, W), dtype=np.int8)
    player = 1
    history = []  # (x, pi, player_at_state)
    move_count = 0

    while True:
        root = Node(board, player)
        pi = mcts_search(root, net, device, sims=sims, c_puct=c_puct, add_dirichlet=True)
        temperature = temp if move_count < temp_moves else 1e-6
        action = select_action_from_pi(pi, temperature)

        # store training sample (before playing action)
        x = encode(board, player, device=torch.device("cpu")).numpy()
        history.append((x, pi.copy(), player))

        # play action
        board = apply_move(board, action, player)
        winner = check_winner(board)
        done = (winner != 0) or is_full(board)
        move_count += 1
        if done:
            # outcome in absolute frame (+1/-1/0)
            outcome = winner  # 0 draw
            samples = []
            for x_i, pi_i, p_i in history:
                z = float(outcome * p_i)  # from player-to-move perspective at that state
                samples.append(Sample(x=x_i, pi=pi_i, z=z))
            return samples

        player = -player

@torch.no_grad()
def eval_vs_random(net: AZNet, device: torch.device, games: int = 20, sims: int = 200) -> float:
    """
    returns winrate vs random (as player +1), draw counts as 0.5
    """
    score = 0.0
    for _ in range(games):
        board = np.zeros((H, W), dtype=np.int8)
        player = 1
        while True:
            if player == 1:
                root = Node(board, player)
                pi = mcts_search(root, net, device, sims=sims, c_puct=1.5, add_dirichlet=False)
                action = int(np.argmax(pi))
            else:
                legal = np.where(legal_moves(board))[0]
                action = int(np.random.choice(legal))
            board = apply_move(board, action, player)
            winner = check_winner(board)
            if winner != 0 or is_full(board):
                if winner == 1: score += 1.0
                elif winner == 0: score += 0.5
                break
            player = -player
    return score / games

# ----------------------------
# Main training loop
# ----------------------------

def train():
    device = torch.device("cpu")
    torch.set_num_threads(8)

    net = AZNet(ch=64, blocks=4).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    buffer = ReplayBuffer(capacity=200_000)

    # Recommended starting params (CPU)
    SIMS = 200          # increase to play stronger
    C_PUCT = 1.5
    GAMES_PER_ITER = 32 # self-play games per iteration
    TRAIN_STEPS = 200   # gradient steps per iteration
    BATCH = 128
    EVAL_EVERY = 5

    for it in range(1, 10_000_000):  # stop manually
        net.eval()
        for _ in range(GAMES_PER_ITER):
            print("Training")
            samples = play_self_game(net, device, sims=SIMS, c_puct=C_PUCT, temp_moves=8, temp=1.0)
            buffer.add_game(samples, augment_mirror=True)

        if len(buffer) < BATCH:
            continue

        net.train()
        pol_losses, val_losses = [], []
        for _ in range(TRAIN_STEPS):
            x, pi_t, z_t = buffer.sample_batch(BATCH, device)
            logits, v = net(x)
            pl, vl, tot = loss_fn(logits, v, pi_t, z_t)

            opt.zero_grad()
            tot.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            pol_losses.append(float(pl.item()))
            val_losses.append(float(vl.item()))

        if it % EVAL_EVERY == 0:
            net.eval()
            wr = eval_vs_random(net, device, games=20, sims=SIMS)
            print(f"[it={it}] buffer={len(buffer)} policy_loss={np.mean(pol_losses):.4f} "
                  f"value_loss={np.mean(val_losses):.4f} winrate_vs_random={wr:.3f}")

if __name__ == "__main__":
    train()
