import theano

from .context import Context

class Gradient(object):
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
    def __init__(self, fn, args):
        # hacky way to get call graph (we could often do it without actually
        # running `fn`)
        self.ctxt = Context()
        self.cost = self.ctxt.call(fn, args)

        self.s_args = [self.ctxt.svars[id(w)] for w in args]
        self.s_cost = self.ctxt.svars[id(self.cost)]

        if self.s_cost.ndim != 0:
            raise NotImplementedError()

        self.g_args = theano.tensor.grad(self.s_cost, self.s_args)

        # compile the function using pure tensor variables for inputs,
        # rather than the shared variables in self.s_args
        s_args2 = [sv.type() for sv in self.s_args]
        self.g_fn = theano.function(s_args2, self.g_args,
                givens=zip(self.s_args, s_args2))

    def __str__(self):
        s = []
        for grad_variable in self.g_fn.maker.env.outputs:
            s += [theano.printing.pprint(grad_variable)]
        return '\n'.join(s)

    def __call__(self, *vals):
        return self.g_fn(*vals)

