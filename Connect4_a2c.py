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
    def __init__(self, state, log_prob, entropy):
        self.state = state
        self.n_state = state
        self.reward = 0.
        self.log_prob = log_prob
        self.done = False
        self.entropy = entropy


class A2C(nn.Module):
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
        fc_layers = [nn.Linear(32*6*7, 128), 
                     nn.ReLU()]
        critic_layer = [nn.Linear(128, 1)]
        actor_layer = [
            nn.Linear(128, 7),
        ]

        # define actor and critic networks
        self.conv = nn.Sequential(*conv_layers).to(self.device)
        self.fc = nn.Sequential(*fc_layers).to(self.device)
        self.critic_model = nn.Sequential(*critic_layer).to(self.device)
        self.actor = nn.Sequential(*actor_layer).to(self.device)
        all_params = (
            list(self.conv.parameters())
                + list(self.fc.parameters())
                + list(self.critic_model.parameters())
                + list(self.actor.parameters())
        )
        # pytorch_total_params = sum(p.numel() for p in self.conv.parameters()) + sum(
        #     p.numel() for p in self.head.parameters()
        # )
        # print(f"Number of parameters -> {pytorch_total_params}")
        self.optim = torch.optim.Adam(all_params, lr=lr)

    def backbone(self, state: torch.Tensor) -> torch.Tensor:
        conv = self.conv(state)
        flat = conv.flatten(start_dim=1)
        fc = self.fc(flat)
        return fc

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.actor(self.backbone(state))
        return logits

    def critic(self, state: torch.Tensor) -> torch.Tensor:
        value = self.critic_model(self.backbone(state))
        return value

    def get_losses(self, memory: list[Memory], gamma: float):
        actor_loss = 0
        critic_loss = 0
        entropy = 0
        for m in memory:
            v = self.critic(m.state.unsqueeze(0)).squeeze()
            with torch.no_grad():
                nv = (
                    self.critic(m.n_state.unsqueeze(0)).squeeze()
                    if not m.done
                    else torch.tensor(0.0, device=self.device)
                )
                target = m.reward + gamma * (-nv)
            # target = torch.tensor(m.reward, device=self.device)
            advantage = (target - v).detach()
            actor_loss += -m.log_prob * advantage
            critic_loss += F.mse_loss(v, target.detach())
            entropy += m.entropy
            # print(
            #     m.reward,
            #     v.item(),
            #     (target - v).item()
            # )
        mean_entropy = entropy / len(memory)
        mean_actor_loss = actor_loss / len(memory)
        actor_loss_kl =  mean_actor_loss - ENTROPY * mean_entropy
        return actor_loss_kl, critic_loss / len(memory)

    def update_parameters(self, actor_loss, critic_loss) -> None:
        self.optim.zero_grad()
        loss = actor_loss + critic_loss * VF_COEFF
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.optim.step()


if torch.cuda.is_available():
    print("Using cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = connect_four_v3.env()  # render_mode="human")
env_manual = connect_four_v3.env(render_mode="human")

EPISODE = 2000
LR = 1e-4  # plus stable
GAMMA = 0.99
ENTROPY = 1e-3
VF_COEFF = 0.8

model = A2C(
    device=device,
    lr=LR,
    n_envs=1,
)
target = A2C(
    device=device,
    lr=LR,
    n_envs=1,
)
target.load_state_dict(model.state_dict())


rolling_length = 20
actor_learning_losses = []
critic_learning_losses = []
first_player_win = []
first_player_losses = []
first_player_deuces = []

change_level_episode = []


def cannonical_state(s):
    x = torch.Tensor(s).to(device)
    x = x.permute(2, 1, 0)
    return x


def select_action_by_value(env, agent, debug=False, train=False, temp=1.0, target=False):
    observation, _, _, _, _ = env.last()
    s = observation["observation"]
    x = cannonical_state(s)

    mask = observation["action_mask"]
    mask = torch.tensor(mask, device=device).bool()
    logits = agent(x.unsqueeze(dim=0)).squeeze(0)
    logits[~mask] = -1e9
    probs = F.softmax(logits/temp, dim=-1)

    if train:
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
    else:
        action = torch.argmax(probs)
        log_prob = None
        entropy = None

    return action.item(), log_prob, entropy


def mesure():
    l_wins = 0
    l_deuce = 0
    l_loss = 0
    for i in range(1):
        env.reset()
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
                action, _, _ = select_action_by_value(env, model, i == 0 and first_play)

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
    if i > 0 and i % 100 == 0:
        print(i)

    env.reset()
    temperature = 0.1 + 0.4 * (max(0, EPISODE-5*i)/EPISODE)

    memory = []
    player = "player_0" if player is None or player == "player_1" else "player_1"
    frame = None
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
            if agent == player:
                first_player_losses.append(1 if reward == -1 else 0)
                first_player_deuces.append(1 if reward == 0 else 0)
                first_player_win.append(1 if reward == 1 else 0)
                if frame is not None:
                    frame.done = True
                    frame.reward = reward
                    memory.append(frame)

            env.step(action)
        else:
            state = observation["observation"]
            state = cannonical_state(state)
            if frame is not None and agent == player:
                frame.n_state = state
                memory.append(frame)

            agent_model = model if agent == player else target
            action, logprobs, entropy = select_action_by_value(
                env, agent_model, train=True, temp=temperature, target=agent != player
            )
            env.step(action)

            if agent == player:
                frame = Memory(
                    state=state,
                    log_prob=logprobs,
                    entropy=entropy
                )
            # frames = augment_d4(frame)
            # for frames_index in range(len(frames)):
            #     memory.append(frames[frames_index])

    R = 0
    for m in reversed(memory):
        R = m.reward + GAMMA * R
        m.reward = R

    actor_loss, critic_loss = model.get_losses(memory=memory, gamma=GAMMA)
    model.update_parameters(actor_loss, critic_loss)

    actor_learning_losses.append(actor_loss.detach().cpu().numpy())
    critic_learning_losses.append(critic_loss.detach().cpu().numpy())

    if i > 0 and i%100==0:
        target.load_state_dict(model.state_dict())
    if i > rolling_length:
        writer.add_scalar("Wins", np.mean(first_player_win[-100:]), i)
        writer.add_scalar("Losses", np.mean(first_player_losses[-100:]), i)
        writer.add_scalar("Deuces", np.mean(first_player_deuces[-100:]), i)
        writer.add_scalar("actor_loss", np.mean(actor_learning_losses[-100:]), i)
        writer.add_scalar("critic_loss", np.mean(critic_learning_losses[-100:]), i)


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
#  loss
axs[1].set_title("Critic loss")
critic_losses_moving_average = (
    np.convolve(
        np.array(critic_learning_losses).flatten(),
        np.ones(rolling_length),
        mode="valid",
    )
    / rolling_length
)
axs[1].plot(critic_losses_moving_average)

#  loss
axs[2].set_title("Actor loss")
actor_losses_moving_average = (
    np.convolve(
        np.array(actor_learning_losses).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[2].plot(actor_losses_moving_average)

plt.tight_layout()
plt.show(block=False)

i_win, i_deuce, i_loss = mesure()
print(f"{i_win},{i_deuce},{i_loss}")

player = None
while True:
    env_manual.reset()
    player = "player_0" if player is None or player == "player_1" else "player_1"
    for agent in env_manual.agent_iter():
        observation, reward, termination, truncation, info = env_manual.last()

        learning_model_round = agent == player
        if termination or truncation:
            action = None
        else:
            if learning_model_round:
                action, _, _ = select_action_by_value(env_manual, model)
            else:
                print("Pick action")
                action = input()
                action = np.array(action, dtype=np.int16)

        env_manual.step(action)

env.close()
env_manual.close()
