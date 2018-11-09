 import numpy as np
import sys, os
sys.path.append(os.pardir)
from collections import OrderedDict

def flatten(x):
    x = np.reshape(x, (784, 1))
    return x


def one_hot_vector(i):
    a = np.zeros(10)
    a[i] = 1
    return a


def sigmoid(t):
    if t.any() <= -34.5:
        return 1e-15
    if t.any() >= 34.5:
        return 1.0 - 1e-15
    return 1/(1+np.exp(-t))


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = np.apply_along_axis(sigmoid, 0, x)
        return sigmoid(x)

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


def relu(x):
    return np.maximum(0, x)


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.W, x) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(self.W.T, dout)
        self.dW = np.dot(dout, self.x.T)
        self.db = np.sum(dout, axis=1)
        self.db = np.reshape(self.db, (np.size(self.db), 1))

        return dx


class Dropout:
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg = True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # output softmax
        self.t = None  # onehotvector

    def forward(self, x, t):
        self.t = t
        self.y = np.apply_along_axis(softmax, 0, x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[1]
        dx = (self.y - self.t) / batch_size

        return dx

# 最適化手法の改良


class Momentum:
    def __init__(self, lr, momentum=0.9):
        self.lr =lr
        self.momentum = momentum
        self.v = 0

    def update(self, params, grads):
        self.v = self.momentum * self.v - self.lr * grads
        return params + self.v


class BatchNorm:
    def __init__(self, gamma, beta, momentum=0.9):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        # テスト時に使用する平均と分散
        self.running_mean = None
        self.running_var = None

        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.dim != 2:
            n, c, h, w = x.shape
            x = x.reshape(n, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flag):
        if self.running_mean is None:
            n, d = x.shape
            self.running_mean = np.zeros(d)
            self.running_var = np.zeros(d)

        if train_flag:
            mu = x.mean(axis=1)
            xc = x - mu
            var = np.mean(xc ** 2, axis=1)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 10e-7))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            n, c, h, w = dout.shape
            dout = dout.reshape(n, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


def cross_entropy_error(y, t):
    batch_size = y.shape[1]
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def init_network():
    network = {}
    d = 28*28
    m = 2000
    c = 10
    b = 100
    network['d'] = d
    network['M'] = m
    network['C'] = c
    network['B'] = b
    network['W1'] = np.random.normal(0, 1/np.sqrt(d), (m, d))
    network['b1'] = np.random.normal(0, 1/np.sqrt(d), (m, 1))
    network['W2'] = np.random.normal(0, 1/np.sqrt(m), (c, m))
    network['b2'] = np.random.normal(0, 1/np.sqrt(m), (c, 1))

    return network


def forward1(network, x):
    w1 = network['W1']
    b1 = network['b1']

    a1 = np.dot(x, w1) + b1
    y1 = sigmoid(a1)
    return y1


def forward2(network, y1):
    w2 = network['W2']
    b2 = network['b2']

    a1 = np.dot(y1, w2) + b2
    y2 = softmax(a1)
    return y2


def predict(network, x):
    w1, w2 = network['W1'], network['W2']
    b1, b2 = network['b1'], network['b2']
    a1 = np.dot(w1, x) + b1
    y1 = sigmoid(a1)
    a2 = np.dot(w2, y1) + b2
    y2 = np.apply_along_axis(softmax, 0, a2)

    return y2


def minibatch():
    b = 100
    arr = np.arange(60000)
    v = [np.random.choice(arr, b, replace=False)]
    v = np.reshape(v, (1, b))
    return v


