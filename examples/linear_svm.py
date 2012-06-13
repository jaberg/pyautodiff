"""
Linear SVM
==========

This example illustrates how a function defined purely by numpy operations can
be minimized directly with a gradient-based solver.

This example is a linear classifier (support vector machine) applied to random
data.

"""

import sys
import numpy as np
from autodiff import fmin_l_bfgs_b

def binary_svm_hinge_loss(weights, bias, x, y, l2_regularization):
    # print x, y
    margin = y * (np.dot(x, weights) + bias)
    loss = np.maximum(0, 1 - margin) ** 2
    l2_cost = 0.5 * l2_regularization * np.dot(weights, weights)
    loss = np.mean(loss) + l2_cost
    return loss


def main():

    # create some fake data
    x = np.random.rand(10, 7)
    y = 2 * (np.random.rand(10) > 0.5) - 1

    def loss_fn(w, b):
        return binary_svm_hinge_loss(w, b, x, y, 1e-4)

    w, b = fmin_l_bfgs_b(loss_fn, [np.zeros(7), np.zeros(())])

    print 'Best-fit SVM:'
    print ' -> cost:', loss_fn(w, b)
    print ' -> weights:', w
    print ' -> bias:', b


if __name__ == '__main__':
    np.random.seed(1)
    sys.exit(main())

