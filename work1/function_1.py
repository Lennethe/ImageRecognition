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


class SGD:
    def __init__(self, lr=0.01):import function_1 as fun
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
mndata = MNIST("/export/home/016/a0160260/le4nn/")
X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0], 28, 28))
Y = np.array(Y)


def predealx(i):
    vx = X[i]
    vx = fun.flatten(vx)
    return vx

def predealy(i):
    vy = Y[i]
    vy = np.apply_along_axis(fun.one_hot_vector, 0, vy)
    return vy


# initialize function

np.random.seed(0)
network = fun.init_network()

iters_num = 100
iter_per_epoch = 100
plotx = []
ploty = []
params = {}
params['W1'] = 0.01 * network['W1']
params['b1'] = np.zeros(np.shape(network['b1']))
params['W2'] = 0.01 * network['W2']
params['b2'] = np.zeros(np.shape(network['b2']))



for j in range(iters_num):
    print(j)
    # making function
    relu = fun.ReLU()
    affine1 = fun.Affine(network['W1'], network['b1'])
    affine2 = fun.Affine(network['W2'], network['b2'])
    sigmoid = fun.Sigmoid()
    softmax = fun.SoftmaxWithLoss()
    optimizer = fun.SGD()

    v = fun.minibatch()
    x = np.apply_along_axis(predealx, 0, v)
    x = np.reshape(x, (network['d'], network['B']))
    t = np.apply_along_axis(predealy, 0, v)

    a1 = affine1.forward(x)
    y1 = sigmoid.forward(a1)

    a2 = affine2.forward(y1)
    res = softmax.forward(a2, t)
    print(res)
    plotx.append(j)
    ploty.append(res)

    grad_ak = softmax.backward(1)
    grad_X = affine2.backward(grad_ak)

    inst2 = sigmoid.backward(y1)
    grad_Y = np.multiply(grad_X, inst2)
    dy = affine1.backward(grad_Y)


    n = 0.01
    network['W1'] = network['W1'] - n * affine1.dW
    network['b1'] = network['b1'] - n * affine1.db
    network['W2'] = network['W2'] - n * affine2.dW
    network['b2'] = network['b2'] - n * affine2.db


np.savez('hs.npz', pW1=network['W1'], pW2=network['W2'], pb1=network['b1'], pb2=network['b2'])

plt.plot(plotx, ploty, label="relu")
plt.legend()
plt.show()
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


def numerical_gardient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range (x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

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


