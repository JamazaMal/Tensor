import tensorflow as tf
import numpy as np

cnt_in = 2
cnt_h1 = 10
cnt_h2 = 15
cnt_out = 1
learning_rate = 0.05


def init_model():
    output = tf.keras.Sequential([
        tf.keras.layers.Dense(cnt_h1, activation="relu", input_dim=cnt_in),
        tf.keras.layers.Dense(cnt_h2, activation="relu"),
        tf.keras.layers.Dense(cnt_out)
    ])
    return output


my_model = init_model()
my_model.compile(loss='mean_squared_error',
                 optimizer='sgd',
                 metrics=['accuracy'])

in_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
out_data = np.array([[0], [1], [1], [0]])
my_model.fit(in_data, out_data, epochs=100, verbose=0, batch_size=2)

print(my_model.predict(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])))





