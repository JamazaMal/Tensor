import AxPipes
import tensorflow as tf
import numpy as np

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

    if parms[0] == 'Input':  # Function to kick off training
        # parms[1::] = List of numerical inputs
        idt.append(list(map(float, parms[1::])))

    if parms[0] == 'Output':  # Function to kick off training
        # parms[1::] = List of numerical outputs
        odt.append(list(map(float, parms[1::])))

    if parms[0] == 'Fit':  # Train the model
        # parms[1] = epochs (number of cycles)
        my_model.fit(np.array(idt), np.array(odt), epochs=int(parms[1]), verbose=0)

    if parms[0] == 'Test':  # Get the result from a single input set
        # parms[1::] = List of numerical outputs
        rout = my_model.predict(np.array([list(map(float, parms[1::]))]))
        AxPipes.respond(str(rout))

    if parms[0] == 'Clear':  # Get the result from a single input set
        # No parms
        idt = list()  # List to hold input data
        odt = list()  # List to hold output data

    if parms[0] == 'GetWeights':
        # parms[1] = Layer number (0 for first layer)
        AxPipes.respond(str(my_model.layers[int(parms[1])].get_weights()))

    sin = AxPipes.receive()
