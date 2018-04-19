import tensorflow as tf
import numpy as np

cnt_in = 2
cnt_hl = 7
cnt_out = 1


my_model = tf.keras.Sequential([
    tf.keras.layers.Dense(cnt_hl, activation="relu", input_dim=cnt_in),
    tf.keras.layers.Dense(cnt_out)
])


my_model.compile(loss='mean_squared_error',
                 optimizer='sgd',
                 metrics=['accuracy'])

in_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
out_data = np.array([[0], [1], [1], [0]])
my_model.fit(in_data, out_data, epochs=1000, verbose=0)

print("---------------------------")
print(my_model.predict(np.array([[0, 0]])))
print(my_model.predict(np.array([[0, 1]])))
print(my_model.predict(np.array([[1, 0]])))
print(my_model.predict(np.array([[1, 1]])))





