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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        conv_model = self.conv
        head_model = self.head
        # conv_model = self.conv
        # head_model = self.head
        conv = conv_model(state)
        mean = conv.mean(dim=(2, 3))
        fc = head_model(mean)
        return fc

    def get_losses(self, memory: list[Memory], critic: nn.Module, gamma, n_step=3):
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

        actor_losses = -log_probs * advantages
        # mean_entropy = entropy / len(memory)
        # mean_actor_loss = actor_loss / len(memory)
        # actor_loss_kl = mean_actor_loss  # - ENTROPY * mean_entropy
        return actor_losses.mean() - ENTROPY * entropies.mean()

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

    def forward(self, state: torch.Tensor, use_target=False) -> torch.Tensor:
        conv_model = self.conv_target if use_target else self.conv
        head_model = self.head_target if use_target else self.head
        # conv_model = self.conv
        # head_model = self.head
        conv = conv_model(state)
        mean = conv.mean(dim=(2, 3))
        fc = head_model(mean)
        return fc

    def get_losses(self, memory: list[Memory], gamma: float, n_step=3) -> torch.Tensor:
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
        loss = F.mse_loss(values, targets, reduction="mean")
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

EPISODE = 5000
LR = 1e-5  # plus stable
GAMMA = 0.9
TD_N = 5
ENTROPY = 1e-3


value = Value(
    device=device,
    lr=LR,
    n_envs=1,
)
policy = Policy(
    device=device,
    lr=LR * 10,
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


def augment_sim(memory: Memory) -> Memory:
    s0 = memory.state
    s1 = memory.n_state
    s0p = torch.flip(input=s0, dims=[1])
    s1p = torch.flip(input=s1, dims=[1])

    return Memory(
        state=s0p,
        n_state=s1p,
        done=memory.done,
        reward=memory.reward,
        action=6 - memory.action,
        mask=memory.mask.flip(0),
    )


player = None

for i in range(EPISODE):
    if i > 0 and i % 50 == 0:
        print(i)

    env.reset(seed=42)

    memory = []
    temperature = 0.1 + 0.4 * (max(0, EPISODE - 2 * i) / EPISODE)
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
            mask = observation["action_mask"]
            mask = torch.tensor(mask, device=device).bool()
            action, log_prob, entropy = select_action_by_value(
                env, policy, train=True, temp=temperature, target=agent != player
            )
            env.step(action)
            observation, reward, termination, truncation, info = env.last()
            env.observe(agent)  # board as agent view
            nstate = observation["observation"]
            x = cannonical_state(nstate)

            frame = Memory(
                state=state,
                n_state=x,
                reward=-reward,
                action=action,
                mask=mask,
                done=termination or truncation,
            )
            memory.append(frame)

            # memory.append(augment_sim(frame))
            # frames = augment_d4(frame)
            # for frames_index in range(len(frames)):
            #     memory.append(frames[frames_index])

    states = [m.n_state for m in memory]
    states = torch.stack(states, dim=0)

    value_loss = value.get_losses(memory=memory, gamma=GAMMA, n_step=TD_N)
    value.update_parameters(value_loss)
    policy_loss = policy.get_losses(
        memory=memory, critic=value, gamma=GAMMA, n_step=TD_N
    )
    policy.update_parameters(policy_loss)

    learning_value_losses.append(value_loss.detach().cpu().numpy())
    learning_policy_losses.append(policy_loss.detach().cpu().numpy())

    if i > rolling_length:
        writer.add_scalar("Wins", np.mean(first_player_win[-100:]), i)
        writer.add_scalar("Losses", np.mean(first_player_losses[-100:]), i)
        writer.add_scalar("Deuces", np.mean(first_player_deuces[-100:]), i)
        writer.add_scalar("value_loss", np.mean(learning_value_losses[-100:]), i)
        writer.add_scalar("policy_loss", np.mean(learning_policy_losses[-100:]), i)


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
