import gymnasium as gym
import random

def pick_sample():
  return random.randint(0, 1)

env = gym.make("CartPole-v1", render_mode="human")
for i in range(1):
  print("start episode {}".format(i))
  done = False
  s, _ = env.reset()
  while not done:
    a = pick_sample()
    s, r, term, trunc, _ = env.step(a)
    done = term or trunc
    print("action: {},  reward: {}".format(a, r))
    print("state: {}, {}, {}, {}".format(s[0], s[1], s[2], s[3]))

env.close()
