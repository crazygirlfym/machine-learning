from model_ranknet.tools import *
import math
import copy
import numpy as np
class NN: #Three layer Neural Network
    def __init__(self, numInputs, numHidden, learningRate=0.001):
        #Inputs: number of input and hidden nodes. Assuming a single output node.
        # +1 for bias node: A node with a constant input of 1. Used to shift the transfer function.
        self.numInputs = numInputs + 1
        self.numHidden = numHidden
        self.numOutput = 1

        # Current activation levels for nodes (in other words, the nodes' output value)

        self.activations_input = np.ones(self.numInputs)
        self.activations_hidden = np.ones(self.numHidden)

        self.activation_output = 1.0 #Assuming a single output.
        self.learning_rate = learningRate

        # create weights
        #A matrix with all weights from input layer to hidden layer
        self.weights_input = np.zeros((self.numInputs, self.numHidden))
        #A list with all weights from hidden layer to the single output neuron.
        self.weights_output = np.zeros(self.numHidden)
        # set them to random vaules
        for i in range(self.numInputs):
            for j in range(self.numHidden):
                self.weights_input[i][j] = random_float(-0.5, 0.5)
        for j in range(self.numHidden):
            self.weights_output[j] = random_float(-0.5, 0.5)

        #Data for the backpropagation step in RankNets.
        #For storing the previous activation levels of all neurons
        self.prevInputActivations = []
        self.prevHiddenActivations = []
        self.prevOutputActivation = 0
        #For storing the previous delta in the output and hidden layer
        self.prevDeltaOutput = 0
        self.prevDeltaHidden = np.zeros(self.numHidden)
        # self.prevDeltaHidden = [0 for i in range(self.numHidden)]
        #For storing the current delta in the same layers
        self.deltaOutput = 0
        self.deltaHidden = np.zeros(self.numHidden)
        # self.deltaHidden = [0 for i in range(self.numHidden)]

    def propagate(self, inputs):
        # print('Propagating input...')
        if len(inputs) != self.numInputs - 1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.prevInputActivations = copy.deepcopy(self.activations_input)
        for i in range(self.numInputs - 1):
            self.activations_input[i] = inputs[i]
        self.activations_input[-1] = 1  # Set bias node to -1.

        # hidden activations
        self.prevHiddenActivations = copy.deepcopy(self.activations_hidden)
        for j in range(self.numHidden):
            sum = 0.0
            for i in range(self.numInputs):
                # print self.ai[i] ," * " , self.wi[i][j]
                sum = sum + self.activations_input[i] * self.weights_input[i][j]
            self.activations_hidden[j] = logFunc(sum)

        # output activations
        self.prevOutputActivation = self.activation_output
        sum = 0.0
        for j in range(self.numHidden):
            sum = sum + self.activations_hidden[j] * self.weights_output[j]
        self.activation_output = logFunc(sum)
        return self.activation_output

    def computeOutputDelta(self):
        '''
        Equations [1-3]
        Updating the delta in the output layer
        '''

        Pab = 1 / (1 + math.exp(-(self.prevOutputActivation - self.activation_output)))
        self.prevDeltaOutput = logFuncDerivative(self.prevOutputActivation) * (1.0 - Pab)
        self.deltaOutput = logFuncDerivative(self.activation_output) * (1.0 - Pab)

    def computeHiddenDelta(self):
        '''
        Equations [4-5]
        Updating the delta values in the hidden layer
        '''

        # Update delta_{A}
        for i in range(self.numHidden):
            self.prevDeltaHidden[i] = logFuncDerivative(self.prevHiddenActivations[i]) * self.weights_output[i] * (
            self.prevDeltaOutput - self.deltaOutput)
        # Update delta_{B}
        for j in range(self.numHidden):
            self.deltaHidden[j] = logFuncDerivative(self.activations_hidden[j]) * self.weights_output[j] * (
            self.prevDeltaOutput - self.deltaOutput)

    def updateWeights(self):
        '''
        Update the weights of the NN
        Equation [6] in the exercise text
        '''

        # Update weights going from the input layer to the output layer
        # Each input node is connected with all nodes in the hidden layer
        for j in range(self.numHidden):
            for i in range(self.numInputs):
                self.weights_input[i][j] = self.weights_input[i][j] + self.learning_rate * (
                self.prevDeltaHidden[j] * self.prevInputActivations[i] - self.deltaHidden[j] * self.activations_input[
                    i])

        # Update weights going from the hidden layer (i) to the output layer (j)
        for i in range(self.numHidden):
            self.weights_output[i] = self.weights_output[i] + self.learning_rate * (
            self.prevDeltaOutput * self.prevHiddenActivations[i] - self.deltaOutput * self.activations_hidden[i])

    # Removed target value(?)
    def backpropagate(self):
        '''
        Backward propagation of error
        1. Compute delta for all weights going from the hidden layer to output layer (Backward pass)
        2. Compute delta for all weights going from the input layer to the hidden layer (Backward pass continued)
        3. Update network weights
        '''

        self.computeOutputDelta()
        self.computeHiddenDelta()
        self.updateWeights()

    def weights(self):
        '''
        Debug: Display network weights
        '''

        print('Input weights:')
        for i in range(self.numInputs):
            print(self.weights_input[i])
        print()
        print('Output weights:')
        print(self.weights_output)