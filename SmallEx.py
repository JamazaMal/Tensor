from random import seed
from random import random
from itertools import repeat
import numpy as np


# Activation function
def activate(inp):  # Relu
    return max(inp, 0)


# Activation function - derivative
def activate_d(inp):
    if inp <= 0:
        return 0
    return 1


# Initialize a network
def initialize_network(n_inputs, n_hidden_layers, n_outputs):
    network = list()
    l1 = n_inputs

    if len(n_hidden_layers) > 0:
        for l2 in n_hidden_layers:
            hidden_layer = [np.array([random() for i in range(l1 + 1)]) for i in range(l2)]
            network.append(hidden_layer)
            l1 = l2
    else:
        l2 = l1

    output_layer = [np.array([random() for i in range(l2+1)]) for i in range(n_outputs)]
    network.append(output_layer)
    return network


def train(network, inputs, expecteds):
    lInputs = inputs
    outnet = [[0]]  # Added 'blank' layer to get outNet to match up with network
    for layer in network:
        lInputs.append(1)
        outputs = []
        for weights in layer:
            outputs.append(activate((lInputs * weights).sum()))
        lInputs = outputs
        outnet.append(outputs)

    first = True
    nw = network
    nw.append(repeat(0))
    nerror = []
    for layer, weights in zip(reversed(outnet), reversed(nw)):
        print(layer, weights)
        if first:
            first = False
            errors = []
            for node, expected in zip(layer, expecteds):
                error = (node-expected)*activate_d(node)
                errors.append(error)
            nerror.append(errors)
        else:
            pass

    print(nerror)
    return()


def predict(network, inputs):
    lInputs = inputs
    for layer in network:
        lInputs.append(1)
        outputs = []
        for weights in layer:
            outputs.append(activate((lInputs * weights).sum()))
        lInputs = outputs
    return(lInputs)




seed(1)

network = initialize_network(2, [3], 2)
# print(network)
# print(predict(network, [1, 1]))
train(network, [1, 1], [1, 1])
