"""
Scipy-based function minimization drivers

"""

import time
import numpy as np
import scipy.optimize.lbfgsb
import theano

from .context import Context


def vector_from_args(args):
    args_sizes = [w.size for w in args]
    x_size = sum(args_sizes)
    x = np.empty(x_size, dtype='float64') # has to be float64 for fmin_l_bfgs_b
    i = 0
    for w in args:
        x[i: i + w.size] = w.flatten()
        i += w.size
    return x


def args_from_vector(x, orig_args):
    # unpack x_opt -> args-like structure `args_opt`
    rval = []
    i = 0
    for w in orig_args:
        rval.append(x[i: i + w.size].reshape(w.shape).astype(w.dtype))
        i += w.size
    return rval


def theano_f_df(fn, args, mode, other_args=(), compile_fn=True):
    """
    Compute gradient wrt args, but not other_args
    """
    # -- inspect bytecode of fn to determine derivative wrt args

    # hacky way to get call graph (we could do it without actually running it)
    ctxt = Context()
    cost = ctxt.call(fn, tuple(args) + tuple(other_args))

    # construct bytecode for f_df() that
    # * unpacks x-> args
    # * computes f, dx
    orig_s_args = [ctxt.svars[id(w)] for w in args]
    args_shapes = [w.shape for w in args]
    x = vector_from_args(args)
    s_x = theano.tensor.vector(dtype=x.dtype)
    s_args = []
    i = 0
    for s_w, w in zip(orig_s_args, args):
        if w.shape:
            s_xi = s_x[i: i + w.size].reshape(w.shape)
        else:
            s_xi = s_x[i]
        s_arg_i = theano.tensor.patternbroadcast(
                s_xi.astype(str(w.dtype)),
                broadcastable=s_w.broadcastable)
        s_args.append(s_arg_i)
        i += w.size


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
    #s_cost = theano.printing.Print('s_cost')(s_cost)
    s_other_args = [ctxt.svars[id(w)] for w in other_args]

    ##theano.printing.debugprint(s_cost)
    ##s_cost = theano.printing.Print('s_cost')(s_cost)

    if not compile_fn:
        return None, locals()
    else:
        if mode is None:
            f_df = theano.function([s_x] + s_other_args, [s_cost, g_x])
        else:
            f_df = theano.function([s_x] + s_other_args, [s_cost, g_x], mode=mode)

        return f_df, locals()


def fmin_l_bfgs_b(fn, args, theano_mode=None, scalar_bounds=None,
        **scipy_kwargs):
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

    f_df, lvars  = theano_f_df(fn, args, mode=theano_mode)

    x = vector_from_args(args)

    if scalar_bounds is not None:
        lb, ub = scalar_bounds
        bounds = np.empty((len(x), 2))
        bounds[:, 0] = lb
        bounds[:, 1] = ub
        if 'bounds' in scipy_kwargs:
            raise TypeError('duplicate argument: bounds')
        scipy_kwargs['bounds'] = bounds
    # pass control to iterative minimizer
    #x_opt, mincost, info_dct = fmin_l_bfgs_b(f_df, x, **fmin_kwargs)
    x_opt, mincost, info_dct = scipy.optimize.lbfgsb.fmin_l_bfgs_b(
            f_df, x, **scipy_kwargs)

    rval = args_from_vector(x_opt, args)
    # XXX: one of the scipy_kwargs says to return more/less info,
    #     and that should be reflected here too.
    return rval

