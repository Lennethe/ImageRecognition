import numpy as np



d = 28*28
M = 4  # instant
C = 10

# N= hyoujunhensa
N = 1

np.random.seed(0)
N = d
W1 = np.random.normal(0, 1/np.sqrt(N), (M, d))
b1 = np.random.normal(0, 1/np.sqrt(N), (M, 1))
N = M
W2 = np.random.normal(0, 1/np.sqrt(N), (C, M))
b2 = np.random.normal(0, 1/np.sqrt(N), (C, 1))
# def predeal


def inputlayer(x):
    return np.reshape(x, (d, 1))


def fullconnectedlayer1(x):
    return sigmoidfun(np.dot(W1, x)+b1)


def fullconnectedlayer2(y):
    return sigmoidfun(np.dot(W2, y)+b2)






def sigmoidfun(v):
    f = lambda x: 1/(1+pow(np.e,x))
    v = np.apply_along_axis(f, 0, v)
    return v



# n = int(input())
# v = [[int(i) for i in input().split()] for _ in range(n)]
# v=sigmoidfunv(v)
# print(v)


def softmaxfun(a):
    alpha = np.amax(a)
    f = lambda x: pow(np.e, x-alpha)
    a1 = np.apply_along_axis(f, 0, a)
    under = np.sum(a1)

    return a1/under


# print(pow(np.e, 0))
# n = int(input())
# v = [[int(i) for i in input().split()] for _ in range(n)]
# v = softmaxfun(v)
# print(v)
# print(np.amax(v))

# def postdeal(a,b)


from mnist import MNIST
mndata = MNIST("/export/home/016/a0160260/le4nn/")
X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0],28,28))
Y = np.array(Y)


def main(i):
    X1 = X[i]
    X2 = inputlayer(X1)
    X3 = fullconnectedlayer1(X2)
    X4 = fullconnectedlayer2(X3)
    X5 = softmaxfun(X4)
    Xres = np.argmax(X5)
    return Xres


# p = 0
# for i in range(in1):
#     if main(i)==Y[i]:
#         p=p+1
# print(p/in1)




# work2

# input is true
def onehotvector(i):
    a = np.zeros(C)
    a[i] = 1
    return a

# n = int(input())
# print(Y[n])
# print(onehotvector(Y[n]))


def deal1(i):
    y1 = X[i]
    y2 = inputlayer(y1)
    y3 = fullconnectedlayer1(y2)
    y4 = fullconnectedlayer2(y3)
    y5 = softmaxfun(y4)
    return y5

def crossentropy(i):
    yk = onehotvector(int(Y[i]))
    yk2 = deal1(i)
    f = lambda x: np.log(x)
    yk2 = np.apply_along_axis(f, 0, yk2)
    return float(((-1)*np.dot(yk,yk2)))

# n = int(input())
# v = [[int(i) for i in input().split()] for _ in range(n)]
# print(v)
# print(np.apply_along_axis(crossentropy, 0, v))


def minibatch(B):
    arr = np.arange(60000)
    arr = [np.random.choice(arr, B, replace=False)]
    crossn =np.apply_along_axis(crossentropy, 0, arr)
    E = np.sum(crossn)/B
    return E

# work


def sigmoidfund(v):
    f = lambda x: 1/(1+pow(np.e,x))
    f1 = lambda x : (1-x)*x
    v = np.apply_along_axis(f, 0, v)
    v = np.apply_along_axis(f1, 0, v)
    return v


