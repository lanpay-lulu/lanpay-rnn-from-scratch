from activation import Tanh
from gate import AddGate, MultiplyGate
import numpy as np


mulGate = MultiplyGate()
addGate = AddGate()
activation = Tanh()

class RNNLayer:
    def forward(self, x, prev_s, U, W, V):
        self.mulu = mulGate.forward(U, x)
        self.mulw = mulGate.forward(W, prev_s)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = activation.forward(self.add)
        self.mulv = mulGate.forward(V, self.s)

    def backward(self, x, prev_s, U, W, V, diff_s, dmulv, forward=True):
        if forward:
            self.forward(x, prev_s, U, W, V)
        dV, dsv = mulGate.backward(V, self.s, dmulv)
        ds = dsv + diff_s
        dadd = activation.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprev_s = mulGate.backward(W, prev_s, dmulw)
        dU, dx = mulGate.backward(U, x, dmulu)
        return (dprev_s, dU, dW, dV)
    
    def backward1(self, x, prev_s, U, W, V, delta1, dmulv, forward=True):
        if forward:
            self.forward(x, prev_s, U, W, V)
        dV, dsv = mulGate.backward(V, self.s, dmulv)
        ds = dsv
        '''
        x (8000, )
        dmulv (8000, )
        add (100, )
        U (100, 8000)
        W (100, 100)
        V (8000, 100)
        delta (100, )
        '''

        m,  = self.add.shape
        #print 'x',x.shape,'  dmulv', dmulv.shape, '   add', self.add.shape, '  V',V.shape, '  W',W.shape

        #z1 = np.dot( np.asmatrix(dmulv), V ) # dL(t)/dh(t)
        z1 = dsv
        #print 'z1 shape', z1.shape
        # h = W(m,n) dot z, so h(t+1) input is n, output is m;
        # So dL/dz(t+1) is m, dz(t+1)/dh(t) is n;
        # So dL(t+1)/dh = dL/dz(t+1) dot W
        z2 = np.dot( np.asmatrix(delta1), W ) # dL(t+1)/dh(t)
        #print 'z2 shape', z2.shape
        dh = np.asarray(np.transpose(z1 + z2)).reshape((m, ))
        #print 'dh shape', dh.shape
        delta = activation.backward(self.add, dh)
        #print 'delta shape', delta.shape
        
        dW = np.dot( np.asmatrix(delta).transpose(), np.asmatrix(prev_s) )
        dU = np.dot( np.asmatrix(delta).transpose(), np.asmatrix(x) )

        dx = np.asarray( np.dot( np.asmatrix(delta), U) ).reshape(x.shape)

        return (delta, dU, dW, dV)
