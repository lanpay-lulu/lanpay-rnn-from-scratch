"""
filename:
    - layer3.py

date:
    - 2017.01.06

description:
    - Implement LSTM layer.
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
    input: x(t), h(t-1), c(t-1) 
    

'''


class LstmGate(object):
    def __init__(self, U, W, B, activation=sig):
        self.U = U
        self.W = W
        self.B = B
        self.activation = activation
        self.init_param_update()
    
    def init_param_update(self):
        self.dU = np.zeros(self.U.shape)
        self.dW = np.zeros(self.W.shape)
        self.dB = np.zeros(self.B.shape)
       
    def update(self, learning_rate):
        self.U -= learning_rate * self.dU
        self.W -= learning_rate * self.dW
        self.B -= learning_rate * self.dB
        self.init_param_update()

    def forward(self, x, prev_s):
        z = mulGate.forward(self.U, x) + mulGate.forward(self.W, prev_s) + self.B
        s = self.activation.forward(z)
        return (z, s)

    def delta_z(self, z):
        return self.activation.backward(z, 1.0)

    def backward(self, delta, x, prev_s): # Note: delta is dz
        self.dU += np.dot( np.asmatrix(delta).transpose(), np.asmatrix(x) ) 
        self.dW += np.dot( np.asmatrix(delta).transpose(), np.asmatrix(prev_s) )
        self.dB += delta
        dx = np.asarray( np.dot( np.asmatrix(delta), self.U) ).reshape(x.shape)
        dprev_s = np.asarray( np.dot( np.asmatrix(delta), self.W) ).reshape(prev_s.shape)
        return (dx, dprev_s) 
        

class LstmLayer(object):
    def __init__(self, ig_param, iv_param, fg_param, og_param, dim):
        self.input_gate = LstmGate(ig_param['U'], ig_param['W'], ig_param['B'])
        self.input_value = LstmGate(iv_param['U'], iv_param['W'], iv_param['B'], tanh)
        self.forget_gate = LstmGate(fg_param['U'], fg_param['W'], fg_param['B'])
        self.output_gate = LstmGate(og_param['U'], og_param['W'], og_param['B'])
        self.gates = (self.input_gate, self.input_value, self.forget_gate, self.output_gate)
        self.dim = dim

    # note: xlist is a list of input, such as a sentence.
    def forward(self, xlist):
        self.T = len(xlist)
        self.xlist = xlist
        self.layers = [] # unit layer
        self.slist = []
        prev_s = np.zeros(self.dim)
        prev_c = np.zeros(self.dim)
        for t in range(self.T):
            layer = LstmUnitLayer()
            x = xlist[t]
            layer.forward(x, prev_s, prev_c, self.gates)
            self.slist.append(layer.s)
            prev_s = layer.s
            prev_c = layer.c
            self.layers.append(layer)
        return self.slist
    
    def bptt(self, dslist):
        T = self.T
        delta = np.zeros(self.dim) # delta is dL / dz(t)
        
        dxlist = []
        forget_gate1_s = np.zeros(self.dim)
        ds1 = np.zeros(self.dim)
        delta_c1 = np.zeros(self.dim)
        for k in range(0, T):
            t = T - k - 1
            if t == 0:
                prev_s = np.zeros(self.dim)
                prev_c = np.zeros(self.dim)
            else:
                prev_s = self.layers[t-1].s
                prev_c = self.layers[t-1].c
            
            x = self.xlist[t]
            ds = dslist[t]
            dx, dprev_s, delta_c1 = self.layers[t].backward( \
                    x, prev_s, ds, ds1, self.gates, forget_gate1_s, delta_c1, prev_c)
            forget_gate1_s = self.layers[t].fg_s
            ds = dx
            ds1 = dprev_s
            dxlist.append(dx)

        return dxlist

    def update(self, learning_rate):
        for gate in self.gates:
            gate.update(learning_rate)


class LstmUnitLayer:
    def forward(self, x, prev_s, prev_c, gates):
        self.x = x
        input_gate, input_value, forget_gate, output_gate = gates
        self.ig_z, self.ig_s = input_gate.forward(x, prev_s) # input gate (z, s)
        self.iv_z, self.iv_s = input_value.forward(x, prev_s) # input value
        self.fg_z, self.fg_s = forget_gate.forward(x, prev_s) # forget gate
        self.og_z, self.og_s = output_gate.forward(x, prev_s) # output gate
        
        self.c = self.ig_s * self.iv_s + self.fg_s * prev_c
        self.cell = tanh.forward(self.c)
        self.s = self.og_s * self.cell 

    def backward(self, 
            x,
            prev_s,
            ds, # delta from next layer
            ds1, # delta from t+1
            gates,
            forget_gate1_s, # next layer forget gate output
            delta_c1,
            prev_c):
        input_gate, input_value, forget_gate, output_gate = gates
        
        delta_s = ds + ds1
        d_og = output_gate.delta_z(self.og_z) * self.cell * delta_s # output gate
        delta_c =   self.og_s * tanh.backward(self.c, 1) * delta_s + \
                    forget_gate1_s * delta_c1 # no peephole

        d_fg =  forget_gate.delta_z(self.fg_z) * prev_c * delta_c  # forget gate
        d_ig =  input_gate.delta_z(self.ig_z) * self.cell * delta_c # input gate
        d_iv = self.ig_s * input_value.delta_z(self.iv_z) * delta_c # input value
        
        d_og_x, d_og_prev_s = output_gate.backward(d_og, x, prev_s)
        d_fg_x, d_fg_prev_s = forget_gate.backward(d_fg, x, prev_s)
        d_ig_x, d_ig_prev_s = input_gate.backward(d_ig, x, prev_s)
        d_iv_x, d_iv_prev_s = input_value.backward(d_iv, x, prev_s)
        
        dx = d_og_x + d_fg_x + d_ig_x + d_iv_x
        dprev_s = d_og_prev_s + d_fg_prev_s + d_ig_prev_s + d_iv_prev_s

        # return new dx(t) and dh(t-1)
        return (dx, dprev_s, delta_c) 

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


