
import gym
import numpy as np

env = gym.make('CartPole-v1')

for i in range(100):
    o = env.reset()
    d = False
    while not d:
        env.render()
        if o[2] > 0:
            o, _, d, _ = env.step(1)
        else:
            if o[2] < 0:
                o, _, d, _ = env.step(0)

