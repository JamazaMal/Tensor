import AxPipes
import tensorflow as tf
import numpy as np


def WeightsToString(_weights):
    sss = ""
    for l in _weights[0]:
        ss = l[0]
        for s in l[1::]:
            ss = "{},{}".format(ss, s)
        if sss == "":
            sss = ss
        else:
            sss = "{}|{}".format(sss, ss)
    l = _weights[1]
    ss = l[0]
    for s in l[1::]:
        ss = "{},{}".format(ss, s)
    return "{}||{}".format(sss, ss)


def StringToWeights(_str):
    _string = ("".join([",{}".format(s) for s in _str]))[1::]
    wts = [np.array([list(map(float, x.split(','))) for x in _string.split('||')[0].split('|')]),
        np.array(list(map(float, _string.split('||')[1].split(','))))]
    return wts


my_model = tf.keras.Sequential()
idt = list()  # List to hold input data
odt = list()  # List to hold output data

sin = AxPipes.receive()
while sin != 'STOP':
    parms = sin.split(',')

    if parms[0] == 'Layer':  # If layer information received
        # parms[1] = activation type
        # parms[2] = no of inputs. (0 omits input_dim)
        # parms[3] = no of outputs
        my_model.add(tf.keras.layers.Dense(int(parms[3]), activation=parms[1], input_dim=int(parms[2])))

    if parms[0] == 'Compile':  # Compile
        # parms[1] = loss function
        # parms[2] = optimizer
        my_model.compile(loss=parms[1], optimizer=parms[2])

    if parms[0] == 'Input':  # Input training data
        # parms[1::] = List of numerical inputs
        idt.append(list(map(float, parms[1::])))

    if parms[0] == 'Output':  # Training data - expected result
        # parms[1::] = List of numerical outputs
        odt.append(list(map(float, parms[1::])))

    if parms[0] == 'Fit':  # Train the model
        # parms[1] = epochs (number of cycles)
        my_model.fit(np.array(idt), np.array(odt), epochs=int(parms[1]), verbose=0)

    if parms[0] == 'Test':  # Get the result from a single input set
        # parms[1::] = List of numerical outputs
        rout = my_model.predict(np.array([list(map(float, parms[1::]))]))
        AxPipes.respond(("".join([",{}".format(s) for s in rout[0]]))[1::])

    if parms[0] == 'Clear':  # Get the result from a single input set
        # No parms
        idt = list()  # List to hold input data
        odt = list()  # List to hold output data

    if parms[0] == 'GetWeights':
        # parms[1] = Layer number (0 for first layer)
        AxPipes.respond(WeightsToString(my_model.layers[int(parms[1])].get_weights()))

    if parms[0] == 'SetWeights':
        # parms[1] = Layer number (0 for first layer)
        # parms[2] = Weights as a string
        my_model.layers[int(parms[1])].set_weights(StringToWeights(parms[2::]))


    sin = AxPipes.receive()
