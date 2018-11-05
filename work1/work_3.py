import function_1 as fun
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

for j in range(iters_num):
    print(j)
    relu = fun.ReLU()
    affine1 = fun.Affine(network['W1'], network['b1'])
    affine2 = fun.Affine(network['W2'], network['b2'])
    sigmoid = fun.Sigmoid()
    softmax = fun.SoftmaxWithLoss()

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