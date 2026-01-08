from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

Player = int  # +1 (X) ou -1 (O)

@dataclass
class StepResult:
    obs: np.ndarray          # (3, H, W) float32 : [current, opponent, player_plane]
    reward: float            # du point de vue du joueur qui vient de jouer
    done: bool
    info: dict

class Gomoku:
    """
    Gomoku minimal:
      - plateau HxW (par défaut 9x9)
      - joueurs: +1 (X) et -1 (O)
      - victoire si >= 5 alignés
      - reward: +1 si le joueur qui joue gagne sur ce coup, 0 sinon (et 0 en match nul)
    """
    def __init__(self, size: int = 9, win_len: int = 5):
        assert size >= win_len >= 2
        self.H = self.W = size
        self.win_len = win_len
        self.board = np.zeros((self.H, self.W), dtype=np.int8)  # 0 vide, +1 X, -1 O
        self.player: Player = +1
        self.done = False
        self.winner: int = 0  # 0 none/draw, +1 X, -1 O
        self.last_move: Optional[Tuple[int, int]] = None

    def reset(self) -> np.ndarray:
        self.board.fill(0)
        self.player = +1
        self.done = False
        self.winner = 0
        self.last_move = None
        return self._obs()

    def legal_moves(self) -> List[int]:
        if self.done:
            return []
        empties = np.argwhere(self.board == 0)
        return [int(r * self.W + c) for r, c in empties]

    def legal_mask(self) -> np.ndarray:
        mask = (self.board.reshape(-1) == 0).astype(np.float32)
        if self.done:
            mask *= 0.0
        return mask  # (H*W,)

    def step(self, action: int) -> StepResult:
        if self.done:
            raise ValueError("Game is done. Call reset().")

        r, c = divmod(int(action), self.W)
        if not (0 <= r < self.H and 0 <= c < self.W):
            raise ValueError(f"Action out of bounds: {action}")
        if self.board[r, c] != 0:
            raise ValueError(f"Illegal move at ({r},{c}).")

        p = self.player
        self.board[r, c] = p
        self.last_move = (r, c)

        # Terminal check
        if self._is_win_from(r, c, p):
            self.done = True
            self.winner = p
            reward = 1.0
        elif np.all(self.board != 0):
            self.done = True
            self.winner = 0
            reward = 0.0
        else:
            reward = 0.0

        # Switch player AFTER reward computed from the mover's perspective
        self.player = -self.player

        info = {"winner": int(self.winner), "last_move": self.last_move}
        return StepResult(self._obs(), reward, self.done, info)

    def render(self) -> None:
        # Affichage console: X = +1, O = -1, . = vide
        chars = {0: ".", +1: "X", -1: "O"}
        print("  " + " ".join([f"{i:2d}" for i in range(self.W)]))
        for r in range(self.H):
            row = " ".join(chars[int(v)] for v in self.board[r])
            print(f"{r:2d} {row}")
        if self.done:
            if self.winner == 0:
                print("== Draw ==")
            else:
                print(f"== Winner: {'X' if self.winner == +1 else 'O'} ==")
        else:
            print(f"To play: {'X' if self.player == +1 else 'O'}")

    # -------- RL helpers --------
    def _obs(self) -> np.ndarray:
        """
        Observation à 3 plans (float32):
          0: pierres du joueur à jouer (current)
          1: pierres de l'adversaire (opponent)
          2: plan constant = +1 si X doit jouer, -1 sinon (broadcast)
        """
        p = self.player
        cur = (self.board == p).astype(np.float32)
        opp = (self.board == -p).astype(np.float32)
        plane = np.full((self.H, self.W), float(p), dtype=np.float32)
        return np.stack([cur, opp, plane], axis=0)

    # -------- Win check (depuis le dernier coup) --------
    def _is_win_from(self, r: int, c: int, p: Player) -> bool:
        # 4 directions: horiz, vert, diag \, diag /
        return (
            self._count_dir(r, c, p, 0, 1) + self._count_dir(r, c, p, 0, -1) - 1 >= self.win_len or
            self._count_dir(r, c, p, 1, 0) + self._count_dir(r, c, p, -1, 0) - 1 >= self.win_len or
            self._count_dir(r, c, p, 1, 1) + self._count_dir(r, c, p, -1, -1) - 1 >= self.win_len or
            self._count_dir(r, c, p, 1, -1) + self._count_dir(r, c, p, -1, 1) - 1 >= self.win_len
        )

    def _count_dir(self, r: int, c: int, p: Player, dr: int, dc: int) -> int:
        cnt = 0
        rr, cc = r, c
        while 0 <= rr < self.H and 0 <= cc < self.W and self.board[rr, cc] == p:
            cnt += 1
            rr += dr
            cc += dc
        return cnt


if __name__ == "__main__":
    env = Gomoku(size=9, win_len=5)
    env.reset()
    # mini démo: place 5 X en ligne (X joue un coup sur deux, on force des coups O ailleurs)
    moves = [(4,0),(0,0),(4,1),(0,1),(4,2),(0,2),(4,3),(0,3),(4,4)]
    for (r,c) in moves:
        a = r*env.W + c
        res = env.step(a)
        env.render()
        print("reward:", res.reward, "done:", res.done, "winner:", res.info["winner"])
        print("-"*40)
        if res.done:
            break
