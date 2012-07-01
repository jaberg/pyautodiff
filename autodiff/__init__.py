"""
autodiff: Automatic differentiation utilities and routines for numpy code.
--------------------------------------------------------------------------

High-level minimization API:

    * fmin_l_bfgs_b
    * fmin_sgd (not implemented)
    * fmin_asgd (not implemented)
    * sgd_iter (not implemented)


Medium-level metaprogramming API:

    * gradient (not implemented)


Low-level implementation API (unstable!):

    * context.Context
    * context.FrameVM

"""

from fmin_scipy import fmin_l_bfgs_b

from fmin_sgd import fmin_sgd

from gradient import Gradient


