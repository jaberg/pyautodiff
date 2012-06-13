"""
Function minimization drivers based on stochastic gradient descent (SGD).

"""

def sgd_iter(fn, args, stepsize):
    raise NotImplementedError()


def fmin_sgd(fn, args, stepsize, n_steps):
    """
    """
    # XXX REFACTOR WITH FMIN

    # STEP 1: inspect bytecode of fn to determine derivative wrt args

    # hacky way to get call graph (we could do it without actually running it)
    ctxt = Context()
    cost = ctxt.call(fn, args)

    # construct bytecode for f_df() that
    # * unpacks x-> args
    # * computes f, dx
    s_args = [ctxt.svars[id(w)] for w in args]
    s_cost = ctxt.svars[id(cost)]

    g_args = theano.tensor.grad(s_cost, s_args)

    # [optional] pass bytecode for g() to numba.translate to compile a faster
    # implementation for the repeated calls that are coming up

    # XXX: hacky current thing does not pass by a proper byte-code optimizer
    # because numba isn't quite there yet. For now we just compile the call
    # graph we already built theano-style.
    update_fn = theano.function([], [s_cost],
            update=[(a, a - stepsize * g) for a, g, in zip(s_args, g_args)],
            )

    # XXX: stopping criterion here
    for i in xrange(10):
        print update_fn()

    return [a.get_value() for a in s_args]


def fmin_asgd(fn, args, stepsize, n_steps):
    raise NotImplementedError()
