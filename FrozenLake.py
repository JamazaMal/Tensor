
import tensorflow as tf
import gym
import numpy as np


cnt_in = 4
cnt_hl = 20
cnt_out = 1


my_model = tf.keras.Sequential([
    tf.keras.layers.Dense(cnt_hl, activation="relu", input_dim=cnt_in),
    tf.keras.layers.Dense(cnt_out, activation="sigmoid")
])

env = gym.make('CartPole-v1')
o = [0 for _ in range(300)]
m = [0 for _ in range(300)]

for i in range(100):
    cnt = 0
    o[cnt] = env.reset()
    m[cnt] = my_model.predict([[o[cnt]]])[0][0]
    d = False
    while not d:
        cnt += 1
        env.render()
        o[cnt], _, d, _ = env.step(int(round(m[cnt])))
        m[cnt] = my_model.predict([[o[cnt]]])[0][0]

