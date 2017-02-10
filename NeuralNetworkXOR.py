import numpy as np


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


class NeuralNetworkXOR:

    def __init__(self, nInput, nHiddenLayer, nOutputLayer):
        self.alpha = 0.5

        self.nInput = nInput
        self.nHiddenLayer = nHiddenLayer
        self.nOutputLayer = nOutputLayer

        self.hiddenWeights = np.random.random((self.nHiddenLayer, self.nInput+1))
        self.outputWeights = np.random.random((self.nOutputLayer, self.nHiddenLayer+1))

        self.hiddenLayerActivation = np.zeros((self.nHiddenLayer, 1), dtype=float)
        self.outputLayerActivation = np.zeros((self.nOutputLayer, 1), dtype=float)

        self.initialOutput = np.zeros((self.nInput+1, 1), dtype=float)
        self.hiddenLayerOutput = np.zeros((self.nHiddenLayer+1, 1), dtype=float)
        self.outputLayerOutput = np.zeros((self.nOutputLayer, 1),  dtype=float)

        self.hiddenLayerDelta = np.zeros(self.nHiddenLayer, dtype=float)
        self.outputLayerDelta = np.zeros(self.nOutputLayer, dtype=float)

    def forward(self, input):
        self.initialOutput[:-1, 0] = input
        self.initialOutput[-1:, 0] = 1.0


        self.hiddenLayerActivation = np.dot(self.hiddenWeights, self.initialOutput)
        self.hiddenLayerOutput[:-1, :] = sigmoid(self.hiddenLayerActivation)

        self.hiddenLayerOutput[-1:, :] = 1.0

        self.outputLayerActivation = np.dot(self.outputWeights, self.hiddenLayerOutput)
        self.outputLayerOutput = sigmoid(self.outputLayerActivation)

    def backward(self, teach):

        error = self.outputLayerOutput - np.array(teach, dtype=float)
        self.outputLayerDelta = (1 - sigmoid(self.outputLayerActivation)) * sigmoid(self.outputLayerActivation) * error
        self.hiddenLayerDelta = (1 - sigmoid(self.outputLayerActivation)) * sigmoid(self.hiddenLayerActivation) * np.dot(self.outputWeights[:, :-1].transpose(), self.outputLayerDelta)

        self.hiddenWeights -= self.alpha * np.dot(self.hiddenLayerDelta, self.initialOutput.transpose())
        self.outputWeights -= self.alpha * np.dot(self.outputLayerDelta, self.hiddenLayerOutput.transpose())

    def getOutput(self):
        return self.outputLayerOutput

if __name__ == '__main__':

    xorSet = [[0, 0], [0, 1], [1, 1], [1, 0]]
    xorTeach = [[0], [1], [0], [1]]

    nn = NeuralNetworkXOR(2, 2, 1)

    count = 0
    while True:
        rnd = np.random.randint(0, 4)

        nn.forward(xorSet[rnd])

        nn.backward(xorTeach[rnd])
        print count, xorSet[rnd], nn.getOutput()[0],
        if nn.getOutput()[0] > 0.8:
            print 'TRUE'
        elif nn.getOutput()[0] < 0.2:
            print 'FALSE'

    print
    count += 1






