
import tensorflow as tf
import numpy

cnt_in = 8
cnt_h1 = 12
cnt_h2 = 8
cnt_out = 1

dataset = numpy.loadtxt("diabetes.csv", delimiter=",")

X = dataset[:, 0:8]
Y = dataset[:, 8]

my_model = tf.keras.Sequential([
    tf.keras.layers.Dense(cnt_h1, activation="relu", input_dim=cnt_in),
    tf.keras.layers.Dense(cnt_h2, activation="relu"),
    tf.keras.layers.Dense(cnt_out, activation="sigmoid")
])

my_model.compile(loss='mean_squared_error',
                 optimizer='adam',
                 metrics=['accuracy'])

my_model.fit(X, Y, epochs=100, batch_size=5, verbose=0)

cnt_tot = 0
cnt_right = 0
for i, res in enumerate(my_model.predict(X)):
    cnt_tot += 1

    if round(res[0]) == Y[i]:
        cnt_right += 1
    else:
        print(i, res[0], Y[i])

print("{}/{} correct {}%".format(cnt_right, cnt_tot, round(cnt_right/cnt_tot*100)))