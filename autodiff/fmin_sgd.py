"""
Function minimization drivers based on stochastic gradient descent (SGD).

"""
import gc
import logging
import sys
import time

import numpy as np

import theano

from .context import Context
from .utils import flat_from_doc, doc_from_flat

logger = logging.getLogger(__name__)
info = logger.info
warn = logger.warn


class FMinSGD(object):
    """
    An iterator implementing the stochastic gradient descent algorithm.
    On each iteration, this function increments each of self.current_args by
    `-stepsize` times its gradient gradient wrt `fn`, and returns the current
    [stochastic] calculation of `fn`.

    """
    def __init__(self, fn, args, streams, stepsize, loops=1,
            theano_mode=None,
            theano_device=None,
            rseed=12345):
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
        theano_device - (API leak) optional string to force cpu/gpu execution
        """
        self.rng = np.random.RandomState(rseed)

        ctxt = Context(device=theano_device)

        s_streams0 = {} # -- symbolic element dictionary
        streams0 = {}  # -- non-symbolic first element dictionary
        _len = sys.maxint
        s_stream_idx = ctxt.shared(np.asarray(0), name='stream_idx')
        s_idxs = ctxt.shared(self.rng.randint(2, size=3), name='idxs')
        s_idx = s_idxs[s_stream_idx]
        for key, stream in streams.items():
            stream0 = stream[0]
            s_stream = ctxt.shared(stream, borrow=True)
            s_stream_i = s_stream[s_idx]
            assert s_stream_i.dtype == str(stream0.dtype)
            if hasattr(stream, 'shape'):
                # -- if stream is a tensor, then all elements have same size
                #    so bake stream0's size into the function.
                ctxt.shadow(stream0, s_stream_i.reshape(stream0.shape))
            else:
                raise NotImplementedError('non ndarray stream', stream)
            streams0[key] = stream0
            s_streams0[key] = s_stream_i
            _len = min(_len, len(stream))

        # -- pass params as args, streams as kwawrgs
        cost = ctxt.call(fn, args, streams0)

        flat_args = flat_from_doc(args)

        s_args = [ctxt.svars[id(w)] for w in flat_args]
        s_cost = ctxt.svars[id(cost)]
        s_stepsize = ctxt.shared(np.asarray(stepsize))

        s_costs = ctxt.shared(np.zeros(3, dtype=s_cost.dtype), name='costs')

        del ctxt
        gc.collect()

        #theano.printing.debugprint(s_cost)

        g_args = theano.tensor.grad(s_cost, s_args,
                warn_type=True,
                disconnected_inputs='warn',
                )

        # -- shared var into which we will write stream entries
        updates = [(a, a - theano.tensor.cast(s_stepsize, a.dtype) * g)
                for a, g, in zip(s_args, g_args)]

        updates += [(s_stream_idx, s_stream_idx + 1)]
        updates += [(s_costs,
            theano.tensor.inc_subtensor(s_costs[s_stream_idx], s_cost))]

        update_fn = theano.function([], [],
                updates=updates,
                mode=theano_mode,
                #profile=1,
                )

        # theano.printing.debugprint(update_fn)

        self.args = args
        self.loops = loops
        self.streams = streams
        self.s_args = s_args
        self.s_cost = s_cost
        self.g_args = g_args
        self.s_streams0 = s_streams0
        self.update_fn = update_fn
        self._len = _len
        self.s_stepsize = s_stepsize
        self.s_stream_idx = s_stream_idx
        self.s_costs = s_costs
        self.s_idxs = s_idxs
        self.ii = 0

    def __iter__(self):
        return self

    def nextN(self, N):
        # Theano's cvm has a really low-overhead direct call
        # interface, which does not permit argument-passing.
        # so we set up all the indexes we want to use in shared
        # variables, and the s_stream_idx iterates over our list
        # of randomly chosen indexes, and fills in the costs into
        # self.s_costs.
        fn = self.update_fn.fn
        _N = min(N, int(self._len * self.loops) - self.ii)
        if _N:
            idxs = self.rng.randint(self._len, size=_N)
            self.s_stream_idx.set_value(0)
            self.s_idxs.set_value(
                    idxs,
                    borrow=True)
            self.s_costs.set_value(
                    np.zeros(_N, dtype=self.s_costs.dtype),
                    borrow=True)
            t0 = time.time()
            try:
                # when using the cvm, there is a special calling form
                # that uses an internal for-loop
                fn(n_calls=_N)
            except TypeError:
                for i in xrange(_N):
                    fn()
            t1 = time.time()
            print 'SGD timing', (t1 - t0), len(idxs)
            rval = list(self.s_costs.get_value())
            self.ii += len(rval)
            return rval
        else:
            return []


    def next(self, N=None):
        rval = self.nextN(1)
        if rval:
            return rval[0]
        else:
            raise StopIteration()

    @property
    def current_args(self):
        vals = [a.get_value() for a in self.s_args]
        rval, pos = doc_from_flat(self.args, vals, 0)
        assert pos == len(vals)
        return rval


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

