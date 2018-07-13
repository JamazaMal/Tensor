import tensorflow as tf
import numpy as np

cnt_in = 2
cnt_h1 = 7
cnt_out = 1

my_model = tf.keras.Sequential()
my_model.add(tf.keras.layers.Dense(cnt_h1, activation="sigmoid", input_dim=cnt_in))
my_model.add(tf.keras.layers.Dense(cnt_out, activation="relu"))

my_model.compile(loss='mean_squared_error',
                 optimizer='sgd',
                 metrics=['accuracy'])

in_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
out_data = np.array([[0], [1], [1], [0]])

my_model.fit(in_data, out_data, epochs=1000, verbose=1)

#my_model.layers[0].set_weights([np.array([[0.5, -0.5], [-0.5,  0.5]]), np.array([0, 0])])
#my_model.layers[1].set_weights([np.array([[.5], [.5]]), np.array([0])])

print("---------------------------")
print(my_model.predict(np.array([[0, 0]]))[0][0])
print(my_model.predict(np.array([[0, 1]]))[0][0])
print(my_model.predict(np.array([[1, 0]]))[0][0])
print(my_model.predict(np.array([[1, 1]]))[0][0])

#print(my_model.predict(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])))

#print(my_model.layers[0].get_weights())
#print(my_model.layers[1].get_weights())

