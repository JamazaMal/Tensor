
import tensorflow as tf
import numpy

# Set some parameters
cnt_in = 8
cnt_h1 = 12
cnt_h2 = 8
cnt_out = 1

# Get the data from file
data_set = numpy.loadtxt("diabetes.csv", delimiter=",")

# Split data into separate training and test sets.
X_train = data_set[0:668, 0:8]
X_test = data_set[668:, 0:8]

Y_train = data_set[0:668, 8]
Y_test = data_set[668:, 8]

# Define model / network
my_model = tf.keras.Sequential([
    tf.keras.layers.Dense(cnt_h1, activation="relu", input_dim=cnt_in),
    tf.keras.layers.Dense(cnt_h2, activation="relu"),
    tf.keras.layers.Dense(cnt_out, activation="sigmoid")
])

# 'Compile' model
my_model.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])

# Train model using training data
my_model.fit(X_train, Y_train, epochs=150, batch_size=100, verbose=0)

# Test models accuracy.
cnt_tot = 0
cnt_right = 0
for i, res in enumerate(my_model.predict(X_test)):
    cnt_tot += 1

    if round(res[0]) == Y_test[i]:
        cnt_right += 1

# Output end result
print("{}/{} correct {}%".format(cnt_right, cnt_tot, round(cnt_right/cnt_tot*100)))