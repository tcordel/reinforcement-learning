import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# Hyperparameters
gamma = 0.99
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(probs, actions, rewards):
    log_probs = -np.log(probs[np.arange(len(actions)), actions])
    return np.sum(log_probs * rewards)

class PolicyPi:
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        self.hidden_weights = np.random.randn(input_dim, hidden_dim) * 0.01
        self.hidden_bias = np.zeros(hidden_dim)
        self.output_weights = np.random.randn(hidden_dim, output_dim) * 0.01
        self.output_bias = np.zeros(output_dim)
        
        self.m_hidden_w = np.zeros_like(self.hidden_weights)
        self.v_hidden_w = np.zeros_like(self.hidden_weights)
        self.m_hidden_b = np.zeros_like(self.hidden_bias)
        self.v_hidden_b = np.zeros_like(self.hidden_bias)
        self.m_output_w = np.zeros_like(self.output_weights)
        self.v_output_w = np.zeros_like(self.output_weights)
        self.m_output_b = np.zeros_like(self.output_bias)
        self.v_output_b = np.zeros_like(self.output_bias)
        self.t = 0

    def forward(self, s):
        hidden = relu(np.dot(s, self.hidden_weights) + self.hidden_bias)
        logits = np.dot(hidden, self.output_weights) + self.output_bias
        return logits, hidden

    def update(self, grad_hidden_w, grad_hidden_b, grad_output_w, grad_output_b):
        self.t += 1
        
        for param, grad, m, v in zip([
            self.hidden_weights, self.hidden_bias,
            self.output_weights, self.output_bias
        ], [
            grad_hidden_w, grad_hidden_b,
            grad_output_w, grad_output_b
        ], [
            self.m_hidden_w, self.m_hidden_b,
            self.m_output_w, self.m_output_b
        ], [
            self.v_hidden_w, self.v_hidden_b,
            self.v_output_w, self.v_output_b
        ]):
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

policy_pi = PolicyPi()

def pick_sample(s):
    logits, _ = policy_pi.forward(s)
    probs = softmax(logits)
    return np.random.choice(len(probs[0]), p=probs[0])

env = gym.make("CartPole-v1")
reward_records = []

for i in range(10000):
    done = False
    states = []
    actions = []
    rewards = []
    s, _ = env.reset()
    while not done:
        states.append(s)
        a = pick_sample(np.expand_dims(s, axis=0))
        s, r, term, trunc, _ = env.step(a)
        done = term or trunc
        actions.append(a)
        rewards.append(r)

    cum_rewards = np.zeros_like(rewards, dtype=np.float32)
    for j in reversed(range(len(rewards))):
        cum_rewards[j] = rewards[j] + (cum_rewards[j+1] * gamma if j+1 < len(rewards) else 0)
    
    # Training step
    states = np.array(states)
    actions = np.array(actions)
    cum_rewards = np.array(cum_rewards)
    logits, hidden = policy_pi.forward(states)
    probs = softmax(logits)
    # loss = cross_entropy_loss(probs, actions, cum_rewards)
    
    # Compute gradients
    dlogits = probs
    dlogits[np.arange(len(actions)), actions] -= 1  # Correctly adjust only selected actions
    dlogits *= cum_rewards[:, None]
    
    grad_output_w = np.dot(hidden.T, dlogits) / len(actions)
    grad_output_b = np.sum(dlogits, axis=0) / len(actions)
    
    dhidden = np.dot(dlogits, policy_pi.output_weights.T)
    dhidden[hidden <= 0] = 0  # ReLU derivative
    
    grad_hidden_w = np.dot(states.T, dhidden) / len(actions)
    grad_hidden_b = np.sum(dhidden, axis=0) / len(actions)
    
    policy_pi.update(grad_hidden_w, grad_hidden_b, grad_output_w, grad_output_b)
    
    print(f"Run episode {i} with rewards {sum(rewards)}", end="\r")
    reward_records.append(sum(rewards))

env.close()

# Generate recent 50 interval average
average_reward = []
for idx in range(len(reward_records)):
    avg_list = reward_records[max(0, idx-49):idx+1]
    average_reward.append(np.mean(avg_list))

# Plot results
plt.plot(reward_records)
plt.plot(average_reward)
plt.show()
