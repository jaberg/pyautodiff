"""
Scipy-based function minimization drivers

"""

import numpy as np
import scipy.optimize.lbfgsb
import theano

from .context import Context


def fmin_l_bfgs_b(fn, args, **scipy_kwargs):
    """
    Return values that minimize Python function `fn(*args)`, by automatically
    differentiating `fn` and applying scipy's fmin_l_bfgs_b routine.

    Parameters
    ----------
    fn: a scalar-valued function of floats and float/complex arguments

    args: a list of floats / complex / ndarrays from from which to start
        optimizing `fn(*args)`
        TODO: support a single float, ndarray, and also tuples, dicts, etc.

    **scipy_kwargs: pass these through to scipy's fmin_l_bfgs_b routine

    """
    # XXX remove algo param, make each algo a separate fmin function

    # STEP 1: inspect bytecode of fn to determine derivative wrt args

    # hacky way to get call graph (we could do it without actually running it)
    ctxt = Context()
    cost = ctxt.call(fn, args)


    # construct bytecode for f_df() that
    # * unpacks x-> args
    # * computes f, dx
    orig_s_args = [ctxt.svars[id(w)] for w in args]
    args_shapes = [w.shape for w in args]
    args_sizes = [w.size for w in args]
    x_size = sum(args_sizes)
    x = np.empty(x_size, dtype='float64') # has to be float64 for fmin_l_bfgs_b
    s_x = theano.tensor.vector(dtype=x.dtype)
    s_args = []
    i = 0
    for w in args:
        x[i: i + w.size] = w.flatten()
        if w.shape:
            s_xi = s_x[i: i + w.size].reshape(w.shape)
        else:
            s_xi = s_x[i]
        i += w.size
        s_args.append(s_xi.astype(str(w.dtype)))

    orig_s_cost = ctxt.svars[id(cost)]
    memo = theano.gof.graph.clone_get_equiv(
            theano.gof.graph.inputs([orig_s_cost]),
            [orig_s_cost],
            memo=dict(zip(orig_s_args, s_args)))
    s_cost = memo[orig_s_cost]
    g_x = theano.tensor.grad(s_cost, s_x)


    # [optional] pass bytecode for g() to numba.translate to compile a faster
    # implementation for the repeated calls that are coming up

    # XXX: hacky current thing does not pass by a proper byte-code optimizer
    # because numba isn't quite there yet. For now we just compile the call
    # graph we already built theano-style.
    f_df = theano.function([s_x], [s_cost, g_x])

    # pass control to iterative minimizer
    #x_opt, mincost, info_dct = fmin_l_bfgs_b(f_df, x, **fmin_kwargs)
    x_opt, mincost, info_dct = scipy.optimize.lbfgsb.fmin_l_bfgs_b(
            f_df, x, **scipy_kwargs)

    # unpack x_opt -> args-like structure `args_opt`
    rval = []
    i = 0
    for w in args:
        rval.append(x_opt[i: i + w.size].reshape(w.shape))
        i += w.size
    # XXX: one of the scipy_kwargs says to return more/less info,
    #     and that should be reflected here too.
    return rval #, mincost, info_dct


