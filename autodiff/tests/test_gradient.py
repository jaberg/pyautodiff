import numpy as np
from autodiff import Gradient

def test_basic():
    g = Gradient(lambda x: ((x + 1) ** 2).sum(), [np.zeros(3)])
    print g
    #f(x) = 2 * x + 2
    print g(np.arange(3))
    #array[2, 4, 6]

