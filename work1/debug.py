import numpy as np
import function_1 as fun

from mnist import MNIST
mndata = MNIST("/export/home/016/a0160260/le4nn/")
X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0],28,28))
Y = np.array(Y)

v = np.load('hs.npz')
network = fun.init_network()
network['W1'] = v['pW1']
network['W2'] = v['pW2']
network['b1'] = v['pb1']
network['b2'] = v['pb2']

count = 0
for i in range(1000):
    x = X[i]
    x = np.reshape(x, (network['d'], 1))
    y_2 = fun.predict(network, x)
    y_2 = np.argmax(y_2)

    y = Y[i]
    print("prediction is", y_2)
    print("answer is", y)

    if y_2 == y:
        count = count+1

print(count/1000)

print(np.shape(network['W1']))
print(np.shape(network['W2']))
print(np.shape(network['b1']))
print(np.shape(network['b2']))
