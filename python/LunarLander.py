import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorNet(nn.Module):
    def __init__(self, l1=32, l2=16):
        super().__init__()

        self.hidden = nn.Linear(8, l1)
        self.hidden1 = nn.Linear(l1, l2)
        self.output = nn.Linear(l2, 4)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        outs = self.hidden1(outs)
        outs = F.relu(outs)
        logits = self.output(outs)
        return logits


class ValueNet(nn.Module):
    def __init__(self, l1=32, l2=16):
        super().__init__()

        self.hidden = nn.Linear(8, l1)
        self.hidden1 = nn.Linear(l1, l2)
        self.output = nn.Linear(l2, 1)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        outs = self.hidden1(outs)
        outs = F.relu(outs)
        value = self.output(outs)
        return value


actor_func = ActorNet().to(device)
value_func = ValueNet().to(device)

all_params = list(actor_func.parameters()) + list(value_func.parameters())

gamma = 0.99  # discount
kl_coeff = 0.20  # weight coefficient for KL-divergence loss
vf_coeff = 0.30  # weight coefficient for value loss
opt = torch.optim.AdamW(all_params, lr=0.005)

# Best config: ${'lr': 0.005, 'l1': 32, 'l2': 16, 'g': 0.99}

# Pick up action and following properties for state (s)
# Return :
#     action (int)       action
#     logits (list[int]) logits defining categorical distribution
#     logprb (float)     log probability
def pick_sample_and_logp(s):
    with torch.no_grad():
        #   --> size : (1, 8)
        s_batch = np.expand_dims(s, axis=0)
        s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
        # Get logits from state
        #   --> size : (1, 4)
        logits = actor_func(s_batch)
        #   --> size : (4)
        logits = logits.squeeze(dim=0)
        # From logits to probabilities
        probs = F.softmax(logits, dim=-1)
        # Pick up action's sample
        #   --> size : (4)
        a = torch.multinomial(probs, num_samples=1)
        #   --> size : ()
        a = a.squeeze(dim=0)
        # Calculate log probability
        logprb = -F.cross_entropy(logits, a, reduction="none")

        # Return
        return a.tolist(), logits.tolist(), logprb.tolist()


# -----
# State
# -----
# [ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ], [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ]
# x, y, vx, vy, θ, vθ, left, right
# ------
# Action
# ------
# 0: do nothing
# 1: fire left orientation engine
# 2: fire main engine
# 3: fire right orientation engine
# action = env.action_space.sample()
env_no_render = gym.make("LunarLander-v3")  # , render_mode="human")
env_with_render = gym.make("LunarLander-v3", render_mode="human")

reward_records = []
for i in range(50000):
    done = False
    states = []
    actions = []
    logits = []
    logprbs = []
    rewards = []
    if i % 1000 == 0 and i > 0:
        env = env_with_render
    else:
        env = env_no_render
    s, _ = env.reset()
    while not done:
        states.append(s)
        a, l, p = pick_sample_and_logp(s)
        s, r, term, trunc, _ = env.step(a)
        done = term or trunc
        actions.append(a)
        logits.append(l)
        logprbs.append(p)
        rewards.append(r)

    #
    # Get cumulative rewards
    #
    cum_rewards = np.zeros_like(rewards)
    reward_len = len(rewards)
    for j in reversed(range(reward_len)):
        cum_rewards[j] = rewards[j] + (
            cum_rewards[j + 1] * gamma if j + 1 < reward_len else 0
        )

    #
    # Train (optimize parameters)
    #
    opt.zero_grad()
    # Convert to tensor
    states = torch.tensor(states, dtype=torch.float).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    logits_old = torch.tensor(logits, dtype=torch.float).to(device)
    logprbs = torch.tensor(logprbs, dtype=torch.float).to(device)
    logprbs = logprbs.unsqueeze(dim=1)
    cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)
    cum_rewards = cum_rewards.unsqueeze(dim=1)
    # Get values and logits with new parameters
    values_new = value_func(states)
    logits_new = actor_func(states)
    # Get advantages
    advantages = cum_rewards - values_new
    ### # Uncomment if you use normalized advantages (see above note)
    ### advantages = (advantages - advantages.mean()) / advantages.std()
    # Calculate P_new / P_old
    logprbs_new = -F.cross_entropy(logits_new, actions, reduction="none")
    logprbs_new = logprbs_new.unsqueeze(dim=1)
    prob_ratio = torch.exp(logprbs_new - logprbs)
    # Calculate KL-div for Categorical distribution (see above)
    l0 = logits_old - torch.amax(logits_old, dim=1, keepdim=True)  # reduce quantity
    l1 = logits_new - torch.amax(logits_new, dim=1, keepdim=True)  # reduce quantity
    l01 = l0 - l1
    e0 = torch.exp(l0)
    e1 = torch.exp(l1)
    e_sum0 = torch.sum(e0, dim=1, keepdim=True)
    e_sum1 = torch.sum(e1, dim=1, keepdim=True)
    e_sum01 = e_sum1 - e_sum0
    p0 = e0 / e_sum0
    kl = torch.sum(
        p0 * (l0 - torch.log(e_sum0) - l1 + torch.log(e_sum1)), dim=1, keepdim=True
    )
    # kl2 = torch.distributions.kl.kl_divergence(logits_old, logits_new)
    # Get value loss
    vf_loss = F.mse_loss(values_new, cum_rewards, reduction="none")
    # Get total loss
    loss = -advantages * prob_ratio + kl * kl_coeff + vf_loss * vf_coeff
    # Optimize
    loss.sum().backward()
    opt.step()

    # Output total rewards in episode (max 500)
    print("Run episode{} with rewards {}".format(i, np.sum(rewards)), end="\r")
    reward_records.append(np.sum(rewards))
env_with_render.close()
env_no_render.close()
