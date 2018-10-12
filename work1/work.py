import numpy as np
import function as fun


def main(i):
    X1 = X[i]
    X2 = fun.inputlayer(X1)
    X3 = fun.fullconnectedlayer1(X2)
    X4 = fun.fullconnectedlayer2(X3)
    X5 = fun.softmaxfun(X4)
    Xres = np.argmax(X5)
    return Xres


# p = 0
# for i in range(in1):
#     if main(i)==Y[i]:
#         p=p+1
# print(p/in1)

in1 = [int(input())]

print("prediction is", end=" ")
print(main(in1))
print("answer is", end=" ")
print(int(Y[in1]))

# For make sure
# import matplotlib.pyplot as plt1

# from pylab import cm
# plt.imshow(X[in1], cmap=cm.gray)
# plt.show()
# print(Y[in1])
