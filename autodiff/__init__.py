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

from fmin_sgd import sgd_iter
from fmin_sgd import fmin_sgd
from fmin_sgd import fmin_asgd


def gradient(fn, args_like=None):
    """
    Returns a function g(*args) that will compute:
        fn(*args), [gradient(x) for x in args]

    in which gradient(x) denotes the derivative in fn(args) wrt each argument.

    When `fn` returns a scalar then the gradients have the same shape as the
    arguments.  When `fn` returns a general ndarray, then the gradients
    have leading dimensions corresponding to the shape of the return value.

    fn - a function that returns a float or ndarray
    args_like - representative arguments, in terms of shape and dtype

    """
    # args must be all float or np.ndarray

    # inspect bytecode of fn to determine derivative wrt args

    # construct bytecode for f_df() that
    # * unpacks x-> args
    # * computes f, dx

    # unpack x_opt -> args-like quantity `args_opt`
    raise NotImplementedError()


