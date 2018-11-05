import function as fun
import numpy as np

from mnist import MNIST
mndata = MNIST("/export/home/016/a0160260/le4nn/")
X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0],28,28))
Y = np.array(Y)

d = 28*28
M = 100  # instant
C = 10

# N= hyoujunhensa
N = 1

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
    W1 = array[0]
    b1 = array[1]
    W2 = array[2]
    b2 = array[3]


    v = fun.minibatch3(B)
    # print(np.shape(v))

    x = np.apply_along_axis(predealx, 0, v)
    x = np.reshape(x, (d, B))


    y = np.apply_along_axis(predealy, 0, v)
    # print(np.sum(y))

    # b1 is really broadcast?
    y_1 = np.dot(W1,x) + b1
    # print(y_1)
    y_1 = np.apply_along_axis(fun.sigmoidfun, 0, y_1)
    # atteruno?
    # print(y_1)

    y_2 = np.dot(W2,y_1) + b2
    y_2 = np.apply_along_axis(fun.softmaxfun, 0, y_2)
    # print(np.shape(y_2))

    grad_ak = (y_2 - y)/B
    # print(grad_ak)

    grad_X = np.dot(W2.T,grad_ak)
    grad_W2 = np.dot(grad_ak, y_1.T)
    grad_b2 = np.sum(grad_ak, axis=1)
    grad_b2 = np.reshape(grad_b2, (C, 1))

    # print(np.shape(grad_X))
    # print(np.shape(grad_W2))
    # print(np.shape(grad_b2))

    # inst1 = np.dot(W2.T, y_2)
    inst2 = np.apply_along_axis(fun.sigmoidfund,1,y_1)
    # print(np.shape(inst1))
    # print(inst2)
    grad_Y = np.multiply(grad_X, inst2)
    # print(grad_Y)

    grad_W1 = np.dot(grad_Y, x.T)
    # print(grad_W1)
    grad_b1 = np.sum(grad_Y, axis=1)
    grad_b1 = np.reshape(grad_b1, (M, 1))

    n = 0.1

    # print(grad_W1)
    # print(grad_b1)
    # print(grad_W2)
    # print(grad_b2)

    # print(np.shape(b1))
    # print(np.shape(grad_b1))
    W1 = W1 - n*grad_W1
    # print(grad_W1)
    W2 = W2 - n*grad_W2
    b1 = b1 - n*grad_b1
    b2 = b2 - n*grad_b2

    ans = [W1, b1, W2, b2]
    return ans


inarray = [sW1, sb1, sW2, sb2]


for i in range(1000):


    inarray = main(inarray)

    W1 = inarray[0]
    b1 = inarray[1]
    W2 = inarray[2]
    b2 = inarray[3]
    # print(np.shape(b1))

    v = fun.minibatch3(B)

    x = np.apply_along_axis(predealx, 0, v)
    x = np.reshape(x, (d, B))

    y = np.apply_along_axis(predealy, 0, v)

    y_1 = np.dot(W1, x) + b1
    y_1 = np.apply_along_axis(fun.sigmoidfun, 0, y_1)

    y_2 = np.dot(W2, y_1) + b2
    y_2 = np.apply_along_axis(fun.softmaxfun, 0, y_2)
    # print(y_2)

    f = lambda x: np.log(x)
    y_2 = np.apply_along_axis(f,0,y_2)
    res = np.multiply(-y, y_2)
    res = np.sum(res)/B

    print(i)
    print(res)
    # print(W1)
    # b1,W2,b2)

np.savez('hs.npz', pW1=W1, pW2=W2, pb1=b1, pb2=b2)
