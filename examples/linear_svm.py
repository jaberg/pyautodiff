"""
Linear SVM
==========

This script fits a linear support vector machine classifier to random data.  It
illustrates how a function defined purely by numpy operations can be minimized
directly with a gradient-based solver.

Program output:

```
ran loss_fn(), returning 1.0
ran loss_fn(), returning 0.722904977725
Best-fit SVM:
 -> cost: 0.722904977725
 -> weights: [-0.61920868 -0.68296249  0.90574115  1.15135323 -0.69036838]
 -> bias: -0.0152805125301
```

"""

import numpy as np
from autodiff import fmin_l_bfgs_b
from functools import partial

np.random.seed(1)

# -- create some fake data
x = np.random.rand(10, 5)
y = 2 * (np.random.rand(10) > 0.5) - 1
l2_regularization = 1e-4

def loss_fn(weights, bias):
    margin = y * (np.dot(x, weights) + bias)
    loss = np.maximum(0, 1 - margin) ** 2
    l2_cost = 0.5 * l2_regularization * np.dot(weights, weights)
    loss = np.mean(loss) + l2_cost
    print 'ran loss_fn(), returning', loss
    return loss

# -- Run loss_fn once to trace computations.
w, b = fmin_l_bfgs_b(loss_fn, [np.zeros(5), np.zeros(())])

# What happened here?
# The computation is repeated many times during the optimization process by
# bytecode *derived* from loss_fn, in which some things (e.g. the print
# statement) have been removed.

# -- run loss_fn as usual
final_loss = loss_fn(w, b)

print 'Best-fit SVM:'
print ' -> cost:', final_loss
print ' -> weights:', w
print ' -> bias:', b

