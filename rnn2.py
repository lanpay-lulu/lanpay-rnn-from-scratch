from datetime import datetime
import numpy as np
import sys
from layer2 import RNNLayer, OutputLayer


class Model:
    def __init__(self, word_dim, hidden_size=[100]):
        self.word_dim = word_dim
        #self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        #self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        #self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))

        self.layer_size = hidden_size
        #self.layer_size = [80, 90]
        self.layer_num = len(self.layer_size)
        self.layers = []
        d1 = word_dim # prev dim
        for hdim in self.layer_size:
            U = np.random.uniform(-np.sqrt(1. / d1), np.sqrt(1. / d1), (hdim, d1))
            W = np.random.uniform(-np.sqrt(1. / hdim), np.sqrt(1. / hdim), (hdim, hdim))
            B = np.random.uniform(-np.sqrt(1. / hdim), np.sqrt(1. / hdim), (hdim, ))
            layer = RNNLayer(U, W, B, hdim) 
            d1 = hdim
            self.layers.append(layer)

        V = np.random.uniform(-np.sqrt(1. / d1), np.sqrt(1. / d1), (word_dim, d1))
        self.output = OutputLayer(V)

    '''
        forward propagation (predicting word probabilities)
        x is one single data, and a batch of data
        for example x = [0, 179, 341, 416], then its y = [179, 341, 416, 1]
    '''
    def forward_propagation(self, xlist):
        input_list = []
        for idx in xlist:
            xx = np.zeros(self.word_dim)
            xx[idx] = 1
            input_list.append(xx)
        for layer in self.layers:
            layer.forward(input_list)
            input_list = layer.slist
        slist = self.output.forward(input_list)
        
        return slist

    def predict(self, x):
        slist = self.forward_propagation(x)
        return [np.argmax(s) for s in slist]
        #return [np.argmax(output.predict(layer.mulv)) for layer in layers]

    def calculate_loss(self, x, y):
        assert len(x) == len(y)
        slist = self.forward_propagation(x)
        loss = 0.0
        for i, s in enumerate(slist):
            loss += self.output.loss(s, y[i])
        return loss / float(len(y))

    def calculate_total_loss(self, X, Y):
        loss = 0.0
        for i in range(len(Y)):
            loss += self.calculate_loss(X[i], Y[i])
        return loss / float(len(Y))

    def bptt(self, x, y, learning_rate):
        assert len(x) == len(y)
        self.forward_propagation(x)
        dV, dslist = self.output.backward(y)
        self.output.update(dV, learning_rate)
        T = len(self.layers)
        for k in range(0, T):
            t = T - k - 1
            layer = self.layers[t]
            dslist, dU, dW, dB = layer.bptt(dslist)
            layer.update(dU, dW, dB, learning_rate)
        
    def sgd_step(self, x, y, learning_rate):
        self.bptt(x, y, learning_rate)

    def train(self, X, Y, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        num_examples_seen = 0
        losses = []
        for epoch in range(nepoch):
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_total_loss(X, Y)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
                # Adjust the learning rate if loss increases
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
            # For each training example...
            for i in range(len(Y)):
                self.sgd_step(X[i], Y[i], learning_rate)
                num_examples_seen += 1
                if i % 20 == 0:
                    print self.layers[0].U[:3, :3], '\n'
        return losses
