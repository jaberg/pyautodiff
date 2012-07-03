"""
Function minimization drivers based on stochastic gradient descent (SGD).

"""
import sys
import time
import numpy as np

import theano

from .context import Context

class FMinSGD(object):
    """
    An iterator implementing the stochastic gradient descent algorithm.
    On each iteration, this function increments each of self.current_args by
    `-stepsize` times its gradient gradient wrt `fn`, and returns the current
    [stochastic] calculation of `fn`.

    """
    def __init__(self, fn, args, stream, stepsize, theano_mode=None):
        """
        fn - a callable taking *(args + (stream[i],))
        args - the arguments of fn, which this function will search
        stream - an iterable (esp. list or ndarray) of objects to calculate
                stochastic gradients.
        stepsize - a multiplier on the negative gradient used for search
        theano_mode - (API leak) how to compile the underlying theano
                function.
        """
        stream0 = stream[0]
        ctxt = Context()
        s_stream0 = theano.shared(stream0)
        if hasattr(stream, 'shape'):
            ctxt.shadow(stream0, s_stream0.reshape(stream0.shape))
        else:
            ctxt.shadow(stream0, s_stream0)

        cost = ctxt.call(fn, tuple(args) + (stream0,))

        s_args = [ctxt.svars[id(w)] for w in args]
        s_cost = ctxt.svars[id(cost)]

        #theano.printing.debugprint(s_cost)

        g_args = theano.tensor.grad(s_cost, s_args)

        # -- shared var into which we will write stream entries
        updates = [(a, a - stepsize * g) for a, g, in zip(s_args, g_args)]
        update_fn = theano.function([], s_cost,
                updates=updates,
                mode=theano_mode,
                )

        # theano.printing.debugprint(update_fn)

        self.s_args = s_args
        self.s_cost = s_cost
        self.g_args = g_args
        self.s_stream0 = s_stream0
        self.update_fn = update_fn
        self.stream = stream
        self.stream_iter = iter(stream)
        self.ii = 0

    def __iter__(self):
        return self

    def nextN(self, N):
        # XXX: stopping criterion here
        fn = self.update_fn
        setval = self.s_stream0.set_value
        stream_iter_next = self.stream_iter.next
        rval = []
        rval_append = rval.append
        try:
            while N:
                # -- write the next element of self.stream into the shared
                #    variable representing our stochastic input
                setval(stream_iter_next(), borrow=True)

                # -- calculate `fn` and perform the arg updates with Theano
                rval_append(fn())
                N -= 1
        except StopIteration:
            self.ii += len(rval)
            raise
        self.ii += len(rval)
        return rval

    def next(self, N=None):
        return self.nextN(1)[0]

    @property
    def current_args(self):
        return [a.get_value() for a in self.s_args]


def fmin_sgd(*args, **kwargs):
    """
    See FMinSGD for documentation. This function creates that object, exhausts
    the iterator, and then returns the final self.current_args values.
    """
    print_interval = kwargs.pop('print_interval', sys.maxint)
    obj = FMinSGD(*args, **kwargs)
    while True:
        try:
            t = time.time()
            vals = obj.nextN(print_interval)
            print 'Value', np.mean(vals), 'time', (time.time() - t)
        except StopIteration:
            break
    return obj.current_args


