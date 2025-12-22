import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import random
from pettingzoo.classic import tictactoe_v3
from torch import cpu, optim
from torch.nn import functional as F


class Memory:
    def __init__(
        self,
        state: torch.Tensor,
        action: int,
        mask: torch.Tensor,
        reward: int,
        done: int,
    ):
        self.state = state
        self.action = action
        self.mask = mask
        self.reward = reward
        self.n_state = torch.ones_like(state)
        self.n_mask = torch.ones_like(mask)
        self.done = done
        pass


class ReplayMemory:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.clear()

    def add(self, item: Memory):
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(item)

    def clear(self):
        self.buffer = []


    def sample(self, sample_size):
        # sampling
        items = self.buffer
        # divide each columns
        states = torch.stack([i.state for i in items], dim=0)
        masks = torch.stack([i.mask for i in items], dim=0)
        n_states = torch.stack([i.n_state for i in items], dim=0)
        n_masks = torch.stack([i.n_mask for i in items], dim=0)
        actions = [i.action for i in items]
        rewards = [i.reward for i in items]
        dones = [i.done for i in items]
        # convert to tensor
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)
        # return result
        return states, masks, actions, rewards, n_states, n_masks, dones

    def length(self):
        return len(self.buffer)


memory = ReplayMemory(buffer_size=10000)


class DQN(nn.Module):
    """
    (Synchronous) Advantage Actor-Critic agent class

    Args:
        n_features: The number of features of the input state.
        n_actions: The number of actions the agent can take.
        device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
                for this code CPU is totally fine).
        critic_lr: The learning rate for the critic network (should usually be larger than the actor_lr).
        actor_lr: The learning rate for the actor network.
        n_envs: The number of environments that run in parallel (on multiple CPUs) to collect experiences.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        device: torch.device,
        lr: float,
        n_envs: int,
    ) -> None:
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        self.n_actions = n_actions

        actor_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(
                32, n_actions
            ),  # estimate action logits (will be fed into a softmax later)
        ]

        # define actor and critic networks
        self.q = nn.Sequential(*actor_layers).to(self.device)
        self.qt = nn.Sequential(*actor_layers).to(self.device)

        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.update_target()

    def update_target(self):
        self.qt.load_state_dict(self.q.state_dict())

    def select_action(self, x: torch.Tensor, mask: torch.Tensor, epsilon: float, learning: bool) -> int:
        if np.random.random() > epsilon:
            if learning:
                q_values = self.q(x)
            else:
                q_values = self.qt(x)
            q_values = q_values.masked_fill(~mask, -np.inf)
            action = torch.argmax(q_values)
            action = action.unsqueeze(dim=0)
            action = action.tolist()[0]
        else:
            action = np.random.choice(
                self.n_actions,
                p=(mask.float().tolist() / np.sum(mask.float().tolist())),
            )
        return action

    def get_losses(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        next_masks: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        with torch.no_grad():
            # compute Q(s_{t+1})                               : size=[batch_size, 2]
            target_vals_for_all_actions = self.qt(next_states)
            target_vals_for_all_actions_masked = target_vals_for_all_actions.masked_fill(
                ~next_masks, -np.inf
            )
            # compute argmax_a Q(s_{t+1})                      : size=[batch_size]
            target_actions = torch.argmax(target_vals_for_all_actions_masked, 1)

            # compute max Q(s_{t+1})                           : size=[batch_size]
            target_actions_one_hot = F.one_hot(target_actions, self.n_actions).float()
            target_vals = torch.sum(
                target_vals_for_all_actions * target_actions_one_hot, 1
            )
            # compute r_t + gamma * (1 - d_t) * max Q(s_{t+1}) : size=[batch_size]
            target_vals_masked = (1.0 - dones) * target_vals
            q_vals1 = rewards + gamma * target_vals_masked

        #
        # Compute q-value
        #
        actions_one_hot = F.one_hot(actions, self.n_actions).float()
        q_vals2 = torch.sum(self.q(states) * actions_one_hot, 1)

        #
        # Get MSE loss and optimize
        #
        loss = F.mse_loss(q_vals1.detach(), q_vals2, reduction="mean")
        return loss

    def update_parameters(self, loss: torch.Tensor) -> None:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


if torch.cuda.is_available():
    print("Using cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = tictactoe_v3.env()  # render_mode="human")
env_manual = tictactoe_v3.env(render_mode="human")

EPISODE = 2000
LR = 1e-5  # plus stable

GAMMA = 1  # ← clé

epsilon = 1.0
EPSILON_DECAY = 1.0 / 500
EPSILON_FINAL = 0
BATCH_SIZE = 64
SAMPLING_SIZE = BATCH_SIZE * 30
TARGET_UPDATE = 2000

model = DQN(
    n_features=18,
    n_actions=9,
    device=device,
    lr=LR,
    n_envs=1,
)


learning_losses = []
losses = []

change_level_episode = []
current_leve_wins = []


def mesure():
    l_wins = 0
    l_deuce = 0
    l_loss = 0
    for i in range(100):
        env.reset(seed=42)
        player = "player_1"

        # if len(current_leve_wins) >= 50 and np.mean(current_leve_wins[-50:]) > 0.65:
        #     change_level_episode.append(i)
        #     opponent.load(model)
        #     current_leve_wins = []
        #
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

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
                with torch.no_grad():
                    mask = observation["action_mask"]
                    state = observation["observation"]
                    mask_values = torch.tensor(mask, dtype=torch.bool, device=device)
                    x = torch.Tensor(state.flatten()).to(device)
                    action = model.select_action(x=x, mask=mask_values, epsilon=0, learning=True)

            env.step(action)
    return (l_wins, l_deuce, l_loss)


i_win, i_deuce, i_loss = mesure()
print(f"{i_win},{i_deuce},{i_loss}")

for i in range(EPISODE):
    env.reset(seed=42)
    player = "player_1" if bool(random.getrandbits(1)) else "player_2"
    player = "player_1"

    items = {}
    # if len(current_leve_wins) >= 50 and np.mean(current_leve_wins[-50:]) > 0.65:
    #     change_level_episode.append(i)
    #     opponent.load(model)
    #     current_leve_wins = []
    #
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        state = observation["observation"]
        mask = observation["action_mask"]
        mask_values = torch.tensor(mask, dtype=torch.bool, device=device)
        x = torch.Tensor(state).to(device)
        x = torch.cat([
            x[:, :, 0].flatten(),
            x[:, :, 1].flatten() * -1
        ])

        if agent in items:
            items[agent].n_state = x
            items[agent].n_mask = mask_values

        learning_model_round = agent == player
        if termination or truncation:
            action = None
            if learning_model_round:
                # items[agent].reward = 0.1 if reward == 0 else reward
                items[agent].reward = reward
                items[agent].done = 1
                memory.add(items[agent])
                losses.append(1 if reward == -1 else 0)
                if reward == 1:
                    current_leve_wins.append(1)
                else:
                    current_leve_wins.append(0)
        else:
            if learning_model_round and agent in items:
                memory.add(items[agent])

            action = model.select_action(x=x, mask=mask_values, epsilon=epsilon, learning=learning_model_round)
            if learning_model_round:
                item = Memory(
                    state=x, action=action, mask=mask_values, reward=0, done=0
                )
                items[agent] = item

        env.step(action)

    # if memory.length() < 5000:
    #     continue

    loss_cuml = None
    states, masks, actions, rewards, n_states, n_masks, dones = memory.sample(
        BATCH_SIZE
    )
    loss = model.get_losses(states, actions, rewards, n_states, n_masks, dones, GAMMA)
    model.update_parameters(loss)
    if loss_cuml is None:
        loss_cuml = loss.detach().cpu().numpy()
    else:
        loss_cuml = loss_cuml + loss.detach().cpu().numpy()

    # Update epsilon
    if epsilon - EPSILON_DECAY >= EPSILON_FINAL:
        epsilon -= EPSILON_DECAY

    # log the losses and entropy
    learning_losses.append(loss_cuml)
    # if len(losses) > 1000 and np.sum(losses[-500:]) <= 0:
    #     break

    # if i % TARGET_UPDATE == 0:
    #     model.update_target()
    #     print(f"update_target ${i}")


rolling_length = 100
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))
fig.suptitle("Training plots for A2C in the TicTacToe environment")

# entropy
axs[0].set_title("Status")
loss_moving_average = (
    np.convolve(np.array(losses), np.ones(rolling_length), mode="valid")
    / rolling_length
)
axs[0].plot(loss_moving_average)

# for i in change_level_episode:
#     axs[0][0].vlines(i, 0, 1)
# axs[0][0].plot(deuces_moving_average)
# axs[0][0].plot(losses_moving_average)
axs[0].set_xlabel("Number of updates")

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

while True:
    env_manual.reset(seed=42)
    player = "player_1"
    for agent in env_manual.agent_iter():
        observation, reward, termination, truncation, info = env_manual.last()

        learning_model_round = agent == player
        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]
            state = observation["observation"]
            if learning_model_round:
                with torch.no_grad():
                    mask_values = torch.tensor(mask, dtype=torch.bool, device=device)
                    x = torch.Tensor(state.flatten()).to(device)
                    action = model.select_action(x=x, mask=mask_values, epsilon=0, learning=True)
            else:
                print("Pick action")
                action = input()
                action = np.array(action, dtype=np.int16)

        env_manual.step(action)

env.close()
env_manual.close()
