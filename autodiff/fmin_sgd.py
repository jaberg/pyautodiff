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
    def __init__(self, fn, args, streams, stepsize, loops=1, theano_mode=None):
        """
        fn - a callable taking *(args + (stream[i],))
        args - the arguments of fn, which this function will search
        stream - a dictionary of iterables (i.e. struct of arrays)
                 These must all have the same length, and FMinSGD will iterate
                 through them jointly, passing the i'th element of each
                 sequence to `fn` to get a gradient estimate.
        stepsize - a multiplier on the negative gradient used for search
        theano_mode - (API leak) how to compile the underlying theano
                function.
        """
        ctxt = Context()

        s_streams0 = {} # -- symbolic element dictionary
        streams0 = {}  # -- non-symbolic first element dictionary
        _len = sys.maxint
        for key in streams:
            stream = streams[key]
            stream0 = stream[0]
            s_stream0 = theano.shared(stream0)
            assert s_stream0.dtype == str(stream0.dtype)
            if hasattr(stream, 'shape'):
                # -- if stream is a tensor, then all elements have same size
                #    so bake stream0's size into the function.
                ctxt.shadow(stream0, s_stream0.reshape(stream0.shape))
            else:
                ctxt.shadow(stream0, s_stream0)
            streams0[key] = stream0
            s_streams0[key] = s_stream0
            _len = min(_len, len(stream))

        # -- pass params as args, streams as kwawrgs
        cost = ctxt.call(fn, args, streams0)

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

        self.loops = loops
        self.streams = streams
        self.s_args = s_args
        self.s_cost = s_cost
        self.g_args = g_args
        self.s_streams0 = s_streams0
        self.update_fn = update_fn
        self._len = _len
        self.reset()

    def reset(self):
        self.ii = 0
        self.loop_jj = 0

    def __iter__(self):
        # XXX: should this call reset?
        return self

    def nextN(self, N):
        fn = self.update_fn
        rval = []
        rval_append = rval.append
        streams = self.streams
        s_streams0 = self.s_streams0
        while N:
            if self.ii >= self._len:
                self.loop_jj += 1
                if self.loop_jj >= self.loops:
                    break
                self.ii = 0
            # -- write the next element of self.stream into the shared
            #    variable representing our stochastic input
            #
            # -- This can be optimized. If the shared var is borrowing
            # memory, then this stepping can be implemented by adding a
            # stride to the data pointer of the shared variable.
            for key in s_streams0:
                s_streams0[key].set_value(
                        streams[key][self.ii], borrow=True)
            # -- calculate `fn` and perform the arg updates with Theano
            rval_append(fn())
            N -= 1
            self.ii += 1
        return rval

    def next(self, N=None):
        rval = self.nextN(1)
        if rval:
            return rval[0]
        else:
            raise StopIteration()

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
        t = time.time()
        vals = obj.nextN(print_interval)
        if vals:
            print 'Value', np.mean(vals), 'time', (time.time() - t)
        else:
            break
    return obj.current_args


