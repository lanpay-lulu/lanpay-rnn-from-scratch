#encoding=utf8
"""
filename:
    - layer2.py

date:
    - 2017.01.04

description:
    - Implement a rnn layer to handle hidden rnn layer.
    - Implement bptt inside this rnn layer.
"""


from activation import Tanh, Sigmoid
from gate import AddGate, MultiplyGate
import numpy as np

mulGate = MultiplyGate()

sig = Sigmoid()
tanh = Tanh()

activation = tanh

'''
For hidden layer cell: 
    - input z(t) = W·x(t) + U·s(t-1) + B
    - output s(t) = activation(z(t))

'''

class RNNLayer(object):
    def __init__(self, U, W, B, dim):
        self.U = U
        self.W = W
        self.B = B
        self.dim = dim
        
    # note: xlist is a list of input, such as a sentence.
    def forward(self, xlist):
        self.T = len(xlist)
        self.layers = [] # rnn unit layer
        self.slist = []
        prev_s = np.zeros(self.dim)
        for t in range(self.T):
            layer = RNNUnitLayer()
            x = xlist[t]
            layer.forward(x, prev_s, self.U, self.W, self.B)
            self.slist.append(layer.s)
            prev_s = layer.s
            self.layers.append(layer)
    
    def bptt(self, dslist):
        T = self.T
        dU = np.zeros(self.U.shape)
        dW = np.zeros(self.W.shape)
        dB = np.zeros(self.B.shape)
        delta = np.zeros(self.dim) # delta is dL / dz(t)
        
        dxlist = []
        for k in range(0, T):
            t = T - k - 1
            if t == 0:
                prev_s = np.zeros(self.dim)
            else:
                prev_s = self.layers[t-1].s
            
            ds = dslist[t]
            delta, dU_t, dW_t, dB_t, dx_t = self.layers[t].backward(prev_s, self.U, self.W, delta, ds)
            dU += dU_t
            dW += dW_t
            dB += dB_t
            dxlist.append(dx_t)

        return (dxlist, dU, dW, dB)

    def update(self, dU, dW, dB, rate):
        self.U -= rate * dU
        self.W -= rate * dW
        self.B -= rate * dB

class RNNUnitLayer:
    def forward(self, x, prev_s, U, W, B):
        self.x = x
        mulu = mulGate.forward(U, x) 
        mulw = mulGate.forward(W, prev_s)
        self.add = mulu + mulw + B
        self.s = activation.forward(self.add) # s is the output. Also use h.

    # delta1 is from t+1 and refers to dL / dz(t+1)
    def backward(self, prev_s, U, W, delta1, ds):
        m,  = self.add.shape
        x = self.x
        #dz = activation.backward(self.add, 1)
        z1 = ds  # dL(t)/dh(t)
        z2 = np.dot( np.asmatrix(delta1), W ) # dL(t+1)/dh(t)
        dh = np.asarray(np.transpose(z1 + z2)).reshape((m, ))
        delta = activation.backward(self.add, dh)

        dW = np.dot( np.asmatrix(delta).transpose(), np.asmatrix(prev_s) )
        dU = np.dot( np.asmatrix(delta).transpose(), np.asmatrix(x) )
        dB = delta
        dx = np.asarray( np.dot( np.asmatrix(delta), U) ).reshape(x.shape)

        return (delta, dU, dW, dB, dx)


class Softmax:
    def fn(self, z): # z is np.array
        e = np.exp(z)
        s = np.sum(e)
        return e/s
    
    def delta(self, z, y):
        s = self.fn(z)
        s[y] -= 1.0
        return s

    def loss(self, s, y):
        return -np.log(s[y])


class OutputLayer:
    def __init__(self, V):
        self.V = V
        self.output = Softmax()

    def loss(self, s, y):
        return self.output.loss(s, y)

    def forward(self, xlist): # x is input
        self.xlist = xlist
        self.zlist = []
        self.slist = []
        for x in xlist:
            z = mulGate.forward(self.V, x)
            s = self.output.fn(z)
            self.zlist.append(z)
            self.slist.append(s)
        return self.slist

    def backward(self, ylist):
        dxlist = []
        dV = np.zeros(self.V.shape)
        for t in range(len(ylist)):
            z = self.zlist[t]
            y = ylist[t]
            x = self.xlist[t]
            delta = self.output.delta(z, y)
            dV_t, dx = mulGate.backward(self.V, x, delta)
            dxlist.append(dx)
            dV += dV_t
        return (dV, dxlist)

    def update(self, dV, rate):
        self.V -= rate * dV


