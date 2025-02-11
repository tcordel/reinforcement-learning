import random
import typing
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from cartpole import CartPole

gamma = 0.99
batch_size = 64
sampling_size = batch_size * 30

epsilon = 1.0
epsilon_decay = epsilon / 3000
epsilon_final = 0.1

class AdamOptimizer:
    def __init__(self, params, lr=0.0005, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0
        self.params = params

    def step(self, grads):
        self.t += 1
        for k in self.params.keys():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class QNet:
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2, lr=0.0005):
        self.weights1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.bias1 = np.zeros(hidden_dim)
        self.weights2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.bias2 = np.zeros(output_dim)
        self.optimizer = AdamOptimizer(self.get_params(), lr)

    def forward(self, s):
        hidden = np.dot(s, self.weights1) + self.bias1
        hidden = np.maximum(0, hidden)  # ReLU activation
        output = np.dot(hidden, self.weights2) + self.bias2
        return output

    def get_params(self):
        return {"w1": self.weights1, "b1": self.bias1, "w2": self.weights2, "b2": self.bias2}

    def update(self, grads):
        self.optimizer.step(grads)

    def copy_from(self, other):
        self.weights1 = np.copy(other.weights1)
        self.bias1 = np.copy(other.bias1)
        self.weights2 = np.copy(other.weights2)
        self.bias2 = np.copy(other.bias2)


def compute_gradients(model, states, actions, rewards, next_states, dones):
    batch_size = len(states)
    
    hidden = np.dot(states, model.weights1) + model.bias1
    hidden = np.maximum(0, hidden)
    q_vals = np.dot(hidden, model.weights2) + model.bias2
    
    next_hidden = np.dot(next_states, model.weights1) + model.bias1
    next_hidden = np.maximum(0, next_hidden)
    next_q_vals = np.dot(next_hidden, model.weights2) + model.bias2
    
    target_q_vals = rewards + gamma * np.max(next_q_vals, axis=1) * (1 - dones)
    q_vals_selected = q_vals[np.arange(batch_size), actions]
    loss_grad = 2 * (q_vals_selected - target_q_vals) / batch_size
    
    grad_w2 = np.dot(hidden.T, np.eye(q_vals.shape[1])[actions] * loss_grad[:, None])
    grad_b2 = np.sum(np.eye(q_vals.shape[1])[actions] * loss_grad[:, None], axis=0)
    
    hidden_grad = np.dot(np.eye(q_vals.shape[1])[actions] * loss_grad[:, None], model.weights2.T)
    hidden_grad[hidden <= 0] = 0  # ReLU gradient
    
    grad_w1 = np.dot(states.T, hidden_grad)
    grad_b1 = np.sum(hidden_grad, axis=0)
    
    return {"w1": grad_w1, "b1": grad_b1, "w2": grad_w2, "b2": grad_b2}

q_model = QNet()
q_target_model = QNet()
q_target_model.copy_from(q_model)

class ReplayMemory:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, item):
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(item)

    def sample(self, sample_size):
        items = random.sample(self.buffer, sample_size)
        states = np.array([i[0] for i in items])
        actions = np.array([i[1] for i in items])
        rewards = np.array([i[2] for i in items])
        n_states = np.array([i[3] for i in items])
        dones = np.array([i[4] for i in items])
        return states, actions, rewards, n_states, dones

    def length(self):
        return len(self.buffer)

memory = ReplayMemory(buffer_size=10000)

def optimize():
    states, actions, rewards, next_states, dones = memory.sample(sampling_size)
    grads = compute_gradients(q_model, states, actions, rewards, next_states, dones)
    q_model.update(grads)

def pick_sample(s, epsilon) -> int:
    if np.random.random() > epsilon:
        q_vals = q_model.forward(s)
        return int(np.argmax(q_vals))
    else:
        return np.random.randint(0, 2)


def evaluate():
    s = env.reset()
    done = False
    total = 0
    while not done:
        a = pick_sample(s, 0.0)
        s_next, r, term, trunc, _ = env.step(a)
        done = term or trunc
        total += r
        s = s_next
    return total


env = CartPole()
reward_records = []

for _ in range(15000):
    done = True
    for _ in range(500):
        if done:
            s = env.reset()
            done = False
            cum_reward = 0

        a = pick_sample(s, epsilon)
        s_next, r, term, trunc, _ = env.step(a)
        done = term or trunc
        memory.add([s, a, r, s_next, float(term)])
        cum_reward += r
        s = s_next

    if memory.length() < 2000:
        continue
    
    optimize()
    total_reward = evaluate()
    reward_records.append(total_reward)
    iteration_num = len(reward_records)
    print(f"Run iteration {iteration_num} rewards {total_reward:.3f} epsilon {epsilon:.5f}", end="\r")

    if iteration_num % 50 == 0:
        q_target_model.copy_from(q_model)
    
    if epsilon - epsilon_decay >= epsilon_final:
        epsilon -= epsilon_decay

print("\nDone")

average_reward = []
for idx in range(len(reward_records)):
    avg_list = np.empty(shape=(1,), dtype=int)
    if idx < 150:
        avg_list = reward_records[: idx + 1]
    else:
        avg_list = reward_records[idx - 149 : idx + 1]
    average_reward.append(np.average(avg_list))
plt.plot(reward_records)
plt.plot(average_reward)
plt.show()
