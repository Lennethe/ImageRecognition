import function as fun
import numpy as np

from mnist import MNIST
mndata = MNIST("/export/home/016/a0160260/le4nn/")
X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0], 28, 28))
Y = np.array(Y)

d = 28*28
M = 10  # instant
C = 10

# N= hyoujunhensa

np.random.seed(0)
N = d
sW1 = np.random.normal(0, 1/np.sqrt(N), (M, d))
sb1 = np.random.normal(0, 1/np.sqrt(N), (M, 1))
N = M
sW2 = np.random.normal(0, 1/np.sqrt(N), (C, M))
sb2 = np.random.normal(0, 1/np.sqrt(N), (C, 1))

# work3

B = 100


def predealx(i):
    x = X[i]
    x = fun.inputlayer(x)
    return x


def predealy(i):
    y = Y[i]
    y = np.apply_along_axis(fun.onehotvector, 0, y)
    return y


def main(array):
    w1 = array[0]
    b1 = array[1]
    w2 = array[2]
    b2 = array[3]

    v = fun.minibatch3(B)
    # print(np.shape(v))

    x = np.apply_along_axis(predealx, 0, v)
    x = np.reshape(x, (d, B))

    y = np.apply_along_axis(predealy, 0, v)
    # print(np.sum(y))

    # b1 is really broadcast?
    y_1 = np.dot(w1, x) + b1
    # print(y_1)
    y_1 = np.apply_along_axis(fun.ReLU, 0, y_1)
    # atteruno?
    # print(y_1)

    y_2 = np.dot(w2, y_1) + b2
    y_2 = np.apply_along_axis(fun.softmaxfun, 0, y_2)
    # print(np.shape(y_2))

    grad_ak = (y_2 - y)/B
    # print(grad_ak)

    grad_x = np.dot(w2.T, grad_ak)
    grad_w2 = np.dot(grad_ak, y_1.T)
    grad_b2 = np.sum(grad_ak, axis=1)
    grad_b2 = np.reshape(grad_b2, (C, 1))

    # print(np.shape(grad_X))
    # print(np.shape(grad_W2))
    # print(np.shape(grad_b2))

    # inst1 = np.dot(W2.T, y_2)
    inst2 = np.apply_along_axis(fun.ReLUdarr, 1, y_1)
    # print(np.shape(inst1))
    # print(inst2)
    grad_y = np.multiply(grad_x, inst2)
    # print(grad_Y)

    grad_w1 = np.dot(grad_y, x.T)
    # print(grad_W1)
    grad_b1 = np.sum(grad_y, axis=1)
    grad_b1 = np.reshape(grad_b1, (M, 1))

    n = 0.1

    # print(grad_W1)
    # print(grad_b1)
    # print(grad_W2)
    # print(grad_b2)

    # print(np.shape(b1))
    # print(np.shape(grad_b1))
    w1 = w1 - n*grad_w1
    # print(grad_W1)
    w2 = w2 - n*grad_w2
    b1 = b1 - n*grad_b1
    b2 = b2 - n*grad_b2

    ans = [w1, b1, w2, b2]
    return ans


inarray = [sW1, sb1, sW2, sb2]
W1_ = inarray[0]
b1_ = inarray[1]
W2_ = inarray[2]
b2_ = inarray[3]


for j in range(300):
    inarray = main(inarray)

    W1_ = inarray[0]
    b1_ = inarray[1]
    W2_ = inarray[2]
    b2_ = inarray[3]
    # print(np.shape(b1))
    v_ = fun.minibatch3(B)

    x_ = np.apply_along_axis(predealx, 0, v_)
    x_ = np.reshape(x_, (d, B))

    y_ = np.apply_along_axis(predealy, 0, v_)

    y_1_ = np.dot(W1_, x_) + b1_
    y_1_ = np.apply_along_axis(fun.ReLU, 0, y_1_)

    y_2_ = np.dot(W2_, y_1_) + b2_
    y_2_ = y_2_ + 1e-7
    y_2_ = np.apply_along_axis(fun.softmaxfun, 0, y_2_)

    y_2_ = np.apply_along_axis(np.log, 0, y_2_)
    res = np.multiply(-y_, y_2_)
    res = np.sum(res)/B

    print(j)
    print(res)
    # print(W1)
    # b1,W2,b2)

np.savez('hs.npz', pW1=W1_, pW2=W2_, pb1=b1_, pb2=b2_)
