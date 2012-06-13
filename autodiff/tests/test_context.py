
import numpy as np
import theano

from autodiff.context import Context


##############################################################################
# numeric routines for testing


def sqr(x):
    y = 2 * x  # irrelevant
    return x * x


def compute_stuff(x):
    # return sum( ((x+1) * 2) ** 2)
    # = sum( 4 * (x+1) ** 2)
    a = x + 1
    b = 2 * a
    c = np.sum(sqr(b))
    print 'HELLO', c   # statement mixed into numeric computations
    return c

def repeat_double(x, N):
    print 'N', N
    for i in range(N):
        x = x + x
        print 'i', i, 'x', x
    assert i == N - 1
    return x


##############################################################################
# helper context-using routines


def grad_fn(ctxt, rval, ival):
    sy = ctxt.svars[id(rval)]
    sx = ctxt.svars[id(ival)]
    dydx = theano.tensor.grad(sy, sx)
    vx = sx.type()
    return theano.function([vx], dydx, givens={sx: vx})


def recalculate_fn(ctxt, rval, ival):
    sy = ctxt.svars[id(rval)]
    sx = ctxt.svars[id(ival)]
    vx = sx.type()
    return theano.function([vx], sy, givens={sx:vx})


##############################################################################
# tests


def test_recalculate():
    x = np.zeros(3)
    c = Context()
    assert compute_stuff(x) == 12
    y = c.call(compute_stuff, (x,))
    assert y == 12

    f = recalculate_fn(c, y, x)
    assert f(x) == 12
    assert f(x + 1) != 12


def test_grad():
    x = np.zeros(3)

    c = Context()
    y = c.call(compute_stuff, (x,))
    assert id(x) in c.svars
    assert id(y) in c.svars
    dy_dx_fn = grad_fn(c, y, x)

    assert np.all(dy_dx_fn(x + 0) == 8)
    assert np.all(dy_dx_fn(x + 1) == 16)
    assert np.all(dy_dx_fn(x + 2) == 24)


def test_loop():
    # test that non-data-dependent loops are unrolled properly

    x = np.zeros(3)
    c = Context()
    y = c.call(repeat_double, (x, 4))

    f = recalculate_fn(c, y, x)
    y2 = f(x + 1)
    assert np.all(y == 0)
    assert np.all(y2 == 16)

