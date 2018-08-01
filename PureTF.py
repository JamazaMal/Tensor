import numpy
from tensorflow.python.keras.datasets import imdb

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=10)

for i in X_train:
    print(i)




