import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from random import random
from pettingzoo.classic import tictactoe_v3
from torch import optim
from torch.nn import functional as F


class A2C(nn.Module):
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
        critic_lr: float,
        actor_lr: float,
        n_envs: int,
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device
        self.n_envs = n_envs

        critic_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # estimate V(s)
        ]

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
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self.actor = nn.Sequential(*actor_layers).to(self.device)

        # define optimizers for actor and critic
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=actor_lr)

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        x = torch.Tensor(x).to(self.device)
        state_values = self.critic(x)  # shape: [n_envs,]
        action_logits_vec = self.actor(x)  # shape: [n_envs, n_actions]
        return (state_values, action_logits_vec)

    def select_action(
        self, x: np.ndarray, mask: torch.tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.

        Args:
            x: A batched vector of states.
            masks: A batched vector of masks for current state

        Returns:
            actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
            state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
        """
        state_values, action_logits = self.forward(x)
        action_logits = action_logits.masked_fill(~mask, -np.inf)
        action_pd = torch.distributions.Categorical(
            logits=action_logits
        )  # implicitly uses softmax
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()
        return actions, action_log_probs, state_values, entropy, action_logits

    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lam: float,
        ent_coef: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
        using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438).

        Args:
            rewards: A tensor with the rewards for each time step in the episode, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions taken at each time step in the episode, with shape [n_steps_per_update, n_envs].
            value_preds: A tensor with the state value predictions for each time step in the episode, with shape [n_steps_per_update, n_envs].
            masks: A tensor with the masks for each time step in the episode, with shape [n_steps_per_update, n_envs].
            gamma: The discount factor.
            lam: The GAE hyperparameter. (lam=1 corresponds to Monte-Carlo sampling with high variance and no bias,
                                          and lam=0 corresponds to normal TD-Learning that has a low variance but is biased
                                          because the estimates are generated by a Neural Net).
            device: The device to run the computations on (e.g. CPU or GPU).

        Returns:
            critic_loss: The critic loss for the minibatch.
            actor_loss: The actor loss for the minibatch.
        """
        T = len(rewards)
        returns = torch.zeros_like(rewards)
        R = 0
        for t in reversed(range(T)):
            R = rewards[t] + gamma * masks[t] * R
            returns[t] = R
        advantages = returns - value_preds
        advantages_actor = torch.clamp(advantages, -1.0, 1.0)
        critic_loss = (returns - value_preds).pow(2).mean()
        actor_loss = (
            -(advantages_actor.detach() * action_log_probs).mean()
            - ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)

    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        """
        Updates the parameters of the actor and critic networks.

        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

    def load(self, other):
        self.critic.load_state_dict(other.critic.state_dict())
        self.actor.load_state_dict(other.actor.state_dict())


if torch.cuda.is_available():
    print("Using cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = tictactoe_v3.env()  # render_mode="human")
env_manual = tictactoe_v3.env(render_mode="human")

EPISODE = 5000
CRITIC_LR = 0.0005
ACTOR_LR = 0.001

GAMMA = 0.9
LAM = 0  # hyperparameter for GAE
ENT_COEF = 0.1  # coefficient for the entropy bonus (to encourage exploration)

model = A2C(
    n_features=18,
    n_actions=9,
    device=device,
    critic_lr=CRITIC_LR,
    actor_lr=ACTOR_LR,
    n_envs=1,
)
opponent = A2C(
    n_features=18,
    n_actions=9,
    device=device,
    critic_lr=CRITIC_LR,
    actor_lr=ACTOR_LR,
    n_envs=1,
)
opponent.load(model)

critic_losses = []
actor_losses = []
entropies = []
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
                    agent_model = model if learning_model_round else opponent
                    mask_values = torch.tensor(mask, dtype=torch.bool, device=device)
                    actions, action_log_probs, state_values, entropy, logits = (
                        agent_model.select_action(x=state.flatten(), mask=mask_values)
                    )
                    probs = F.softmax(logits, dim=-1)
                    action = np.argmax(probs.detach().numpy())

            env.step(action)
    return (l_wins, l_deuce, l_loss)

i_win, i_deuce, i_loss = mesure()
print(f"{i_win},{i_deuce},{i_loss}")

for i in range(EPISODE):
    env.reset(seed=42)
    player = "player_1"

    np_ep_rewards = []
    np_ep_action_log_probs = []
    np_ep_value_preds = []
    np_ep_entropy = []
    np_ep_masks = []

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
                np_ep_rewards[len(np_ep_rewards) - 1] = reward
                losses.append(1 if reward == -1 else 0)
                if reward == 1:
                    current_leve_wins.append(1)
                else:
                    current_leve_wins.append(0)
        else:
            mask = observation["action_mask"]
            state = observation["observation"]
            agent_model = model if learning_model_round else opponent
            mask_values = torch.tensor(mask, dtype=torch.bool, device=device)
            actions, action_log_probs, state_values, entropy, logits = (
                agent_model.select_action(x=state.flatten(), mask=mask_values)
            )
            action = actions.cpu().detach().numpy()
            if learning_model_round:
                np_ep_rewards.append(0)
                np_ep_action_log_probs.append(action_log_probs)
                np_ep_value_preds.append(state_values)
                np_ep_masks.append(1)
                np_ep_entropy.append(entropy)

        env.step(action)

    ep_rewards = torch.tensor(data=np_ep_rewards, dtype=torch.int16, device=device)
    ep_action_log_probs = torch.stack(np_ep_action_log_probs, dim=0)
    ep_value_preds = torch.stack(np_ep_value_preds, dim=0)
    ep_entropy = torch.stack(np_ep_entropy, dim=0)
    ep_masks = torch.tensor(data=np_ep_masks, dtype=torch.int16, device=device)
    critic_loss, actor_loss = model.get_losses(
        ep_rewards,
        ep_action_log_probs,
        ep_value_preds,
        ep_entropy,
        ep_masks,
        GAMMA,
        LAM,
        ENT_COEF,
        device,
    )

    model.update_parameters(critic_loss, actor_loss)

    # log the losses and entropy
    critic_losses.append(critic_loss.detach().cpu().numpy())
    actor_losses.append(actor_loss.detach().cpu().numpy())
    entropies.append(ep_entropy.detach().mean().cpu().numpy())


rolling_length = 100
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
fig.suptitle("Training plots for A2C in the TicTacToe environment")

# entropy
axs[0][0].set_title("Status")
loss_moving_average = (
    np.convolve(np.array(losses), np.ones(rolling_length), mode="valid")
    / rolling_length
)
axs[0][0].plot(loss_moving_average)

for i in change_level_episode:
    axs[0][0].vlines(i, 0, 1)
# axs[0][0].plot(deuces_moving_average)
# axs[0][0].plot(losses_moving_average)
axs[0][0].set_xlabel("Number of updates")

# entropy
axs[1][0].set_title("Entropy")
entropy_moving_average = (
    np.convolve(np.array(entropies), np.ones(rolling_length), mode="valid")
    / rolling_length
)
axs[1][0].plot(entropy_moving_average)
axs[1][0].set_xlabel("Number of updates")


# critic loss
axs[0][1].set_title("Critic Loss")
critic_losses_moving_average = (
    np.convolve(
        np.array(critic_losses).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0][1].plot(critic_losses_moving_average)
axs[0][1].set_xlabel("Number of updates")


# actor loss
axs[1][1].set_title("Actor Loss")
actor_losses_moving_average = (
    np.convolve(np.array(actor_losses).flatten(), np.ones(rolling_length), mode="valid")
    / rolling_length
)
axs[1][1].plot(actor_losses_moving_average)
axs[1][1].set_xlabel("Number of updates")

plt.tight_layout()
plt.show(block = False)

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
                agent_model = model
                mask_values = torch.tensor(mask, dtype=torch.bool, device=device)
                actions, action_log_probs, state_values, entropy, logits = (
                    agent_model.select_action(x=state.flatten(), mask=mask_values)
                )
                probs = F.softmax(logits, dim=-1)
                action = np.argmax(probs.detach().numpy())
            else:
                print('Pick action')
                action = input()
                action = np.array(action, dtype=np.int16)

        env_manual.step(action)

env.close()
env_manual.close()
