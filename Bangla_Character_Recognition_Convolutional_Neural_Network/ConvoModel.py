from ConvolutionLayer import Convolution
from ReLU import ReLU
from MaxPoolingLayer import MaxPooling
from FlatteningLayer import Flattening
from FullyConnectedLayer import FullyConnected
from SoftMaxLayer import Softmax


class ConvoModel:

    def __int__(self, learning_rate=0.01):
        self.input_channels = 1
        self.output_channels = 6
        self.filter_dim = 12
        self.stride = 3
        self.padding = 1
        self.learning_rate = learning_rate

        self.convolution = Convolution(self.input_channels, self.output_channels, self.filter_dim, self.stride, self.padding)
        self.relu = ReLU()
        self.max_pooling = MaxPooling(psool_size=2, stride=2)
        self.flatten = Flattening()
        self.fully_connected = FullyConnected(output_units=10, learning_rate=self.learning_rate)
        self.softmax = Softmax()


    def forward(self, x):
        x = self.convolution.forward(x)
        x = self.relu.forward(x)
        x = self.max_pooling.forward(x)
        x = self.flatten.forward(x)
        x = self.fully_connected.forward(x)
        x = self.softmax.forward(x)

        print("ConvoModel forward: ", x.shape)
        return x

    def backward(self, delta):
        delta = self.softmax.backward(delta)
        delta = self.fully_connected.backward(delta)
        delta = self.flatten.backward(delta)
        delta = self.max_pooling.backward(delta)
        delta = self.relu.backward(delta)
        delta = self.convolution.backward(delta)

        return delta
