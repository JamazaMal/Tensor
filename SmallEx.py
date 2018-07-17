import tensorflow as tf
import numpy as np



ss = "1.1,2.2,3.3|4.4,5.5,6.6|1.0,2.0,3.0||7.7,8.8,9.9"

w1 = [np.array([list(map(float, x.split(','))) for x in ss.split('||')[0].split('|')]),
      np.array(list(map(float, ss.split('||')[1].split(','))))]

print(w1)

# exit()

cnt_in = 2
cnt_h1 = 3
cnt_out = 1

my_model = tf.keras.Sequential()
my_model.add(tf.keras.layers.Dense(cnt_h1, activation="sigmoid", input_dim=cnt_in))
my_model.add(tf.keras.layers.Dense(cnt_out, activation="sigmoid", input_dim=cnt_h1))

my_model.compile(loss='mean_squared_error',
                 optimizer='adamax')

#my_model.compile(loss='mean_squared_error',
#                 optimizer='sgd',
#                 metrics=['accuracy'])

idt = []
s = ['Input', '0', '0']
idt.append(list(map(float, s[1::])))
s = ['Input', '0', '1']
idt.append(list(map(float, s[1::])))
s = ['Input', '1', '0']
idt.append(list(map(float, s[1::])))
s = ['Input', '1', '1']
idt.append(list(map(float, s[1::])))

in_data = np.array(idt)
out_data = np.array([[0], [1], [1], [0]])

# my_model.layers[0].set_weights([np.array([[0.5, -0.5], [-0.5,  0.5]]), np.array([0, 0])])
# my_model.layers[1].set_weights([np.array([[.5], [.5]]), np.array([0])])


my_model.fit(in_data, out_data, epochs=10, verbose=0)
print("---------------------------")
# print(my_model.predict(np.array([[0, 0]]))[0][0])
# print(my_model.predict(np.array([[0, 1]]))[0][0])
# print(my_model.predict(np.array([[1, 0]]))[0][0])
# print(my_model.predict(np.array([[1, 1]]))[0][0])

#print(my_model.predict(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])))

zzz = my_model.layers[0].get_weights()
sss = ""
for l in zzz[0]:
    ss = l[0]
    for s in l[1::]:
        ss = "{},{}".format(ss, s)
    if sss == "":
        sss = ss
    else:
        sss = "{}|{}".format(sss, ss)
l = zzz[1]
ss = l[0]
for s in l[1::]:
    ss = "{},{}".format(ss, s)
sss = "{}||{}".format(sss, ss)

print(sss)

print(zzz)


