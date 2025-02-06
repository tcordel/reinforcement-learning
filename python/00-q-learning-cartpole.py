import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt

from cartpole import CartPole


new_observation_shape = (20, 20, 20, 20)

bins = []
for i in range(4):
    offset = 4
    if (i == 0):
        offset = 1
    if (i==2):
        offset = 12 * 2 * math.pi / 360
    item = np.linspace(
        -offset,
        offset,
        num=new_observation_shape[i],
        endpoint=False)
    item = np.delete(item, 0)
    bins.append(item)
    print(bins[i])

# define function to convert to discrete state
def get_discrete_state(s):
    new_s = []
    for i in range(4):
        new_s.append(np.digitize(s[i], bins[i]))
    return new_s

q_table = np.zeros(new_observation_shape + (2,))
q_table.shape


gamma = 0.99
alpha = 0.1
epsilon = 1
epsilon_decay = epsilon / 4000

# pick up action from q-table with greedy exploration
def pick_sample(s, epsilon) -> int:
    # get optimal action,
    # but with greedy exploration (to prevent picking up same values in the first stage)
    if np.random.random() > epsilon:
        a = np.argmax(q_table[tuple(s)]).item()
    else:
        a = np.random.randint(0, 2)
    return a

reward_records = []
for i in range(6000):
    # Run episode till done
    done = False
    total_reward = 0

    env = CartPole()
    s = env.reset()
    s_dis = get_discrete_state(s)
    while not done:
        a = pick_sample(s_dis, epsilon)
        s, r, term, trunc, _ = env.step(a)
        done = term or trunc
        s_dis_next = get_discrete_state(s)

        # Update Q-Table
        maxQ = np.max(q_table[tuple(s_dis_next)])
        q_table[tuple(s_dis)][a] += alpha * (r + gamma * maxQ - q_table[tuple(s_dis)][a])

        s_dis = s_dis_next
        total_reward += r

    # Update epsilon for each episode
    if epsilon - epsilon_decay >= 0:
        epsilon -= epsilon_decay

    # Record total rewards in episode (max 500)
    print("Run episode{} with rewards {}".format(i, total_reward), end="\r")
    reward_records.append(total_reward)

print("\nDone")

# Generate recent 50 interval average
average_reward = []
for idx in range(len(reward_records)):
    avg_list = np.empty(shape=(1,), dtype=int)
    if idx < 50:
        avg_list = reward_records[:idx+1]
    else:
        avg_list = reward_records[idx-49:idx+1]
    average_reward.append(np.average(avg_list))
# Plot
plt.plot(reward_records)
plt.plot(average_reward)
plt.show()
