"""
Function minimization drivers based on stochastic gradient descent (SGD).

"""

import theano

from .fmin_scipy import vector_from_args, args_from_vector
from .fmin_scipy import theano_f_df

class FMinSGD(object):
    def __init__(self, fn, args, stream, stepsize, theano_mode=None):
        stream0 = stream[0]
        f_df, l_vars = theano_f_df(fn, args, mode=theano_mode,
                other_args=(stream0,),
                compile_fn=False)

        s_args = l_vars['orig_s_args']
        s_cost = l_vars['orig_s_cost']
        g_args = theano.tensor.grad(s_cost, s_args)

        # -- shared var into which we will write stream entries
        s_stream0 = l_vars['ctxt'].svars[id(stream0)]

        update_fn = theano.function([],
                [s_cost],
                updates=[(a, a - stepsize * g) for a, g, in zip(s_args, g_args)],
                )

        self.s_args = s_args
        self.s_cost = s_cost
        self.g_args = g_args
        self.s_stream0 = s_stream0
        self.update_fn = update_fn
        self.stream = stream
        self.stream_iter = iter(stream)

    def __iter__(self):
        return self

    def next(self, N=1):
        # XXX: stopping criterion here
        fn = self.update_fn
        setval = self.s_stream0.set_value
        stream_iter_next = self.stream_iter.next
        while N:
            setval(stream_iter_next(), borrow=True)
            rval = fn()
            N -= 1
        return rval

    @property
    def current_args(self):
        return [a.get_value() for a in self.s_args]



def fmin_sgd(*args, **kwargs):
    """
    """
    obj = FMinSGD(*args, **kwargs)
    while True:
        try:
            val = obj.next(100)
            # print val
        except StopIteration:
            break
    return obj.current_args


