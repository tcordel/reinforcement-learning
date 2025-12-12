from collections import deque
import numpy as np
import numpy.random as random
import torch
import torch.nn as nn
from torch.nn import functional as F
from enum import Enum
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        r = int(a / 3)
        c = int(a % 3)
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


class ActorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(18, 18)
        self.output = nn.Linear(18, 9)

    def forward(self, s):
        outs = s.view(-1, 18)
        outs = self.l1(outs)
        outs = F.relu(outs)
        outs = self.output(outs)
        return outs


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(18, 18)
        self.output = nn.Linear(18, 1)

    def forward(self, s):
        outs = s.view(-1, 18)
        outs = self.l1(outs)
        outs = F.relu(outs)
        outs = self.output(outs)
        return outs


def idx_to_rc(idx):
    return divmod(idx, 3)


def rc_to_idx(r, c):
    return r * 3 + c


def apply_transform(
    state_t: torch.Tensor, mask_flat: np.ndarray, action: int, t_id: int
):
    s = state_t.clone()
    mask2d = mask_flat.reshape(3, 3).copy()  # numpy 9x9
    r, c = idx_to_rc(action)

    if t_id == 0:  # identity
        pass
    elif t_id == 1:  # rot90
        s = torch.rot90(s, 1, (1, 2))
        mask2d = np.rot90(mask2d, 1)
        r, c = c, 8 - r
    elif t_id == 2:  # rot180
        s = torch.rot90(s, 2, (1, 2))
        mask2d = np.rot90(mask2d, 2)
        r, c = 2 - r, 2 - c
    elif t_id == 3:  # rot270
        s = torch.rot90(s, 3, (1, 2))
        mask2d = np.rot90(mask2d, 3)
        r, c = 2 - c, r
    elif t_id == 4:  # flip horizontal
        s = torch.flip(s, [1])
        mask2d = np.flipud(mask2d)
        r = 2 - r
    elif t_id == 5:  # flip vertical
        s = torch.flip(s, [2])
        mask2d = np.fliplr(mask2d)
        c = 2 - c
    elif t_id == 6:  # flip + rot90
        s = torch.flip(s, [1])
        s = torch.rot90(s, 1, (1, 2))
        mask2d = np.rot90(np.flipud(mask2d), 1)
        r, c = c, r
    elif t_id == 7:  # flip + rot270
        s = torch.flip(s, [1])
        s = torch.rot90(s, 3, (1, 2))
        mask2d = np.rot90(np.flipud(mask2d), 3)
        r, c = 2 - c, 2 - r

    new_action = rc_to_idx(r, c)
    new_mask_flat = mask2d.reshape(-1).astype(int).copy()
    return s.cpu().numpy().copy(), new_mask_flat, int(new_action)


def augment_state(state_np, mask_flat, action):
    # state_np: numpy (2,9,9)
    s_t = torch.tensor(state_np, dtype=torch.float)
    out = []
    for t in range(8):
        s_aug, m_aug, a_aug = apply_transform(s_t, mask_flat, action, t)
        out.append((s_aug, m_aug, a_aug))
    return out


if __name__ == "__main__":
    # instantiate nets properly
    actor_func = ActorNet().to(device)
    opp_func = ActorNet().to(device)
    opp_func.load_state_dict(actor_func.state_dict())  # copy weights
    legacy = ActorNet().to(device)
    legacy.load_state_dict(actor_func.state_dict())

    value_func = ValueNet().to(device)

    batch_size = 64
    gamma = 0.99
    kl_coeff = 1  # weight coefficient for KL-divergence loss
    vf_coeff = 0.10  # weight coefficient for value loss

    # pick up action with above distribution policy_pi
    def pick_sample(s, mask, fun):
        with torch.no_grad():
            s_batch = np.expand_dims(s, axis=0)
            s_batch = torch.tensor(s_batch, dtype=torch.float, device=device)
            logits = fun(s_batch)  # shape (1,9)
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(
                0
            )  # (1,9)
            # remplacer invalides par -inf
            large_neg = -1e9
            logits = logits.masked_fill(~mask_t, large_neg)
            logits = logits.squeeze(0)  # (9,)
            # si toutes masquées (improbable), choisir aléatoire
            if mask.sum() == 0:
                raise Exception("Should not be here")
            probs = F.softmax(logits, dim=-1)
            a = torch.multinomial(probs, num_samples=1)
            a = a.squeeze(dim=0)
            # logprb = -F.cross_entropy(logits, a, reduction="none")
            return a.tolist()  # , logits.tolist(), logprb.tolist()

    all_params = list(actor_func.parameters()) + list(value_func.parameters())
    opt = torch.optim.AdamW(all_params, lr=0.05)
    env = InnerGame()

    evolve = False
    LEVEL = 5000
    evolve_counter = 0
    for level in range(LEVEL):
        # identifiants pour tes réseaux
        PLAYER_CUR = "current"
        PLAYER_OPP = "opponent"

        reward_records = []
        reward_history = deque(maxlen=200)
        win_history = deque(maxlen=200)
        print("Level -> {}".format(level))
        if evolve:
            evolve_counter = evolve_counter + 1
            evolve = False
            opp_func.load_state_dict(actor_func.state_dict())
            torch.save(actor_func.state_dict(), f"./simple-kl-{level}.pth")
            evolve_counter = 0
        else:
            print("Not evolving !!! ", flush=True)

        for i in range(100000):
            done = False
            myId = random.randint(1, 2)
            env.reset()
            me = myId == 1
            transitions = []
            s, status, mask = env.state(myId if me else (3 - myId))
            while not done:
                fun = actor_func if me else opp_func
                mask = env.valid_mask()
                a = pick_sample(s, mask, fun)
                # store mask for training only for 'me' states (we train policy of 'me' only)
                if me:
                    transitions.append([(s, mask, a)])
                env.step(a, myId if me else (3 - myId))
                me = not me
                s, status, mask = env.state(myId if me else (3 - myId))
                done = status != Status.PENDING

            # final reward to last move of player 1
            r = (
                1.0
                if (status == Status.P1 if myId == 1 else status == Status.P2)
                else (
                    0.0
                    if (status == Status.P2 if myId == 1 else status == Status.P1)
                    else 0.1
                )
            )

            elo_score_curr = 0.5 if r == 0.1 else r
            win_history.append(r)

            # cumulative rewards
            reward_len = len(transitions)
            cum_rewards = np.zeros(reward_len, dtype=float)
            for j in reversed(range(reward_len)):
                cum_rewards[j] = 0 + (
                    cum_rewards[j + 1] * gamma if j + 1 < reward_len else r
                )

            reward_records.append(r)
            reward_history.append(r)

            buffer = []
            with torch.no_grad():
                for j in range(reward_len):
                    augmented_state = transitions[j]
                    for k in range(len(augmented_state)):
                        s, m, a = augmented_state[k]
                        s_batch = np.expand_dims(s, axis=0)
                        s_batch = torch.tensor(
                            s_batch, dtype=torch.float, device=device
                        )
                        logits_v = actor_func(s_batch)  # shape (1,9)
                        mask_t = torch.tensor(
                            m, dtype=torch.bool, device=device
                        ).unsqueeze(0)  # (1,9)
                        # remplacer invalides par -inf
                        large_neg = -1e9
                        logits_v = logits_v.masked_fill(~mask_t, large_neg)
                        logits_v = logits_v.squeeze(0)  # (9,)
                        # si toutes masquées (improbable), choisir aléatoire
                        if m.sum() == 0:
                            raise Exception("Should not be here")
                        probs = F.softmax(logits_v, dim=-1)
                        a = torch.multinomial(probs, num_samples=1)
                        a = a.squeeze(dim=0)
                        logprb = -F.cross_entropy(logits_v, a, reduction="none")

                        buffer.append(
                            (
                                s,
                                a,
                                cum_rewards[j],
                                m,
                                logits_v.tolist(),
                                logprb.tolist(),
                            )
                        )

            mmm = np.mean(win_history)
            vf_loss_history = deque(maxlen=200)
            pi_loss_history = deque(maxlen=200)
            kl_history = deque(maxlen=200)

            states, actions, cum_rewards, masks, logits_old, logprbs = zip(*buffer)
            states = torch.tensor(list(states), dtype=torch.float).to(device)
            actions = torch.tensor(list(actions), dtype=torch.int64).to(device)
            cum_rewards = torch.tensor(list(cum_rewards), dtype=torch.float).to(device)
            masks = torch.tensor(list(masks), dtype=torch.bool).to(device)
            logits_old = torch.tensor(list(logits_old), dtype=torch.float).to(device)
            logprbs = torch.tensor(list(logprbs), dtype=torch.float).to(device)
            # Convert to tensor
            # states = torch.tensor(states, dtype=torch.float).to(device)
            # actions = torch.tensor(actions, dtype=torch.int64).to(device)
            # masks = torch.tensor(masks, dtype=torch.bool).to(device)
            # logits_old = torch.tensor(logits, dtype=torch.float).to(device)
            # logprbs = torch.tensor(logprbs, dtype=torch.float).to(device)
            # logprbs = logprbs.unsqueeze(dim=1)
            # cum_rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            # cum_rewards = cum_rewards.unsqueeze(dim=1)
            # Get values and logits with new paraked_fill(~masks, large_neg)
            values_new = value_func(states)
            logits_new = actor_func(states)

            large_neg = -1e9
            logits_new = logits_new.masked_fill(~masks, large_neg)

            # Get advantages
            advantages = cum_rewards - values_new
            ### # Uncomment if you use normalized advantages (see above note)
            ### advantages = (advantages - advantages.mean()) / advantages.std()
            # Calculate P_new / P_old
            logprbs_new = -F.cross_entropy(logits_new, actions, reduction="none")
            logprbs_new = logprbs_new.unsqueeze(dim=1)
            prob_ratio = torch.exp(logprbs_new - logprbs)
            # Calculate KL-div for Categorical distribution (see above)
            l0 = logits_old - torch.amax(
                logits_old, dim=1, keepdim=True
            )  # reduce quantity
            l1 = logits_new - torch.amax(
                logits_new, dim=1, keepdim=True
            )  # reduce quantity
            e0 = torch.exp(l0)
            e1 = torch.exp(l1)
            e_sum0 = torch.sum(e0, dim=1, keepdim=True)
            e_sum1 = torch.sum(e1, dim=1, keepdim=True)
            p0 = e0 / e_sum0
            kl = torch.sum(
                p0 * (l0 - torch.log(e_sum0) - l1 + torch.log(e_sum1)),
                dim=1,
                keepdim=True,
            )
            # Get value loss
            vf_loss = F.mse_loss(values_new, cum_rewards, reduction="none")
            pi_loss = -advantages * prob_ratio
            # Get total loss
            loss = pi_loss + kl * kl_coeff + vf_loss * vf_coeff
            # Optimize

            opt.zero_grad()
            loss.sum().backward()
            opt.step()
            vf_loss_history.append(vf_loss.mean().item())
            pi_loss_history.append(pi_loss.mean().item())
            kl_history.append(kl.mean().item())

            if len(reward_records) >= 100:
                diff_elo = np.sum(reward_records[-100:])

                # if i > 0 and i % 10 == 0:
                # print(
                #     "Level {} processing !!! , last 100 avg reward {:.3f}".format(
                #         level, np.mean(reward_records[-100:]) if reward_records else 0.0
                #     )
                # )
                #
                print(
                    "episode {} with avg reward {:.3f}".format(
                        i, np.average(reward_records[-100:])
                    ),
                    flush=True,
                )
                if diff_elo > 60:
                    evolve = True
                    print(
                        "Stop at episode {} with avg reward {:.3f}".format(
                            i, np.average(reward_records[-100:])
                        ),
                        flush=True,
                    )
                    break

        # print(
        #     f"[Level {level}] avg reward={np.mean(reward_history):.3f}, "
        #     f"winrate={np.mean(win_history) * 100:.1f}%, "
        #     f"vf_loss={np.mean(vf_loss_history):.4f}, "
        #     f"pi_loss={np.mean(pi_loss_history):.4f}",
        #     f"kl={np.mean(kl_history):.4f}",
        # )

    torch.save(actor_func.state_dict(), "./simple-kl-final.pth")
    print("\nDone")
