
import tensorflow as tf
import numpy

dataset = numpy.loadtxt("pima-indians-diabetes.txt", delimiter=",")

X = dataset[:, 0:8]
Y = dataset[:, 8]


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(12, input_dim=8, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#  model.fit(X, Y, epochs=150, batch_size=10,  verbose=2)

predictions = model.predict(X)
rounded = [round(x[0]) for x in predictions]
print(X[1])
print(rounded)
