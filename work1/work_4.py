import numpy as np
import sys, os
sys.path.append(os.pardir)
from function_1 import *
from collections import OrderedDict


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight__init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight__init_std * np.random.randn(input_size, hidden_size)


