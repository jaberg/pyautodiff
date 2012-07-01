"""
Example of how to use byte-code execution technique to trace accesses to numpy arrays.

This file demonstrates two applications of this technique:
* optimize numpy computations for repeated calling
* provide automatic differentiation of procedural code

"""

import __builtin__
from functools import partial
import inspect
import logging; logger = logging.getLogger(__name__)
import opcode
import os
import sys
import trace


import numpy as np
import theano

from .utils import itercode

from scipy.optimize.lbfgsb import fmin_l_bfgs_b

logger.setLevel(logging.INFO)

# Opcode help: http://docs.python.org/library/dis.html

class Unassigned(object): """Unassigned value"""


class FrameVM(object):
    """
    A Class for evaluating a code block of CPython bytecode,
    and tracking accesses to numpy arrays.

    """
    def __init__(self, watcher, func):
        #print 'FrameVM', func
        self.watcher = watcher
        self.func = func
        self.stack = []
        self._locals = None
        self._myglobals = None
        self.code_iter = None
        self.print_ops = False
        self.print_stack = False

        # self.varnames = self.fco.co_varnames
        # self.costr = func.func_code.co_code
        # self.argnames = self.fco.co_varnames[:self.fco.co_argcount]

    def add_shadow(self, x):
        # -- We cannot safely set up shadow variables that are aliased to
        #    memory that is visible to the running program, unless that
        #    program can guarantee that all views of that memory are
        #    immutable.
        borrow = False
        if isinstance(x, (int, float)):
            if type(x) is int and 0 <= x < 256:
                raise Exception('cannot shadow low integer constants')
            s_x = theano.shared(np.asarray(x), borrow=borrow)
        elif x.dtype == bool:
            print >> sys.stderr, "Warning: Theano has no bool, upgrading to uint8"
            s_x = theano.shared(x.astype('uint8'), borrow=borrow)
        else:
            s_x = theano.shared(x, borrow=borrow)
        self.watcher.shadow(x, s_x)

    def ensure_shadow(self, x):
        # CPython re-uses ids for low integers, so we can't shadow them
        if type(x) is int and 0 <= x < 256:
            # It is admitedly a misnomer that ensure_shadow() does not in fact
            # create an svars entry for id(x)...  not sure how to deal with
            # that.
            assert id(x) not in self.watcher.svars
            return theano.tensor.as_tensor_variable(x)

        if id(x) not in self.watcher.svars:
            self.add_shadow(x)
        return self.watcher.svars[id(x)]

    def call(self, args, kwargs):

        func = self.func
        func_code = self.func.func_code
        co_varnames = self.func.func_code.co_varnames
        co_argcount = self.func.func_code.co_argcount

        self._myglobals = {}
        self._locals = [Unassigned] * len(co_varnames)

        _locals = self._locals
        if hasattr(func, 'im_self'):
            _locals[0] = func.im_self
            bind_varnames = co_varnames[1:]
            bind_offset = 1
            if id(func.im_self) in self.watchers.svars:
                raise NotImplementedError('bound method on shadowed var: %s' %
                        func.__name__)
        else:
            bind_varnames = co_varnames
            bind_offset = 0

        for name in func_code.co_names:
            #print 'name', name
            try:
                self._myglobals[name] = func.func_globals[name]
            except KeyError:
                try:
                    self._myglobals[name] = __builtin__.__getattribute__(name)
                except AttributeError:
                    #print 'WARNING: name lookup failed', name
                    pass

        extra_args_ok = bool(func_code.co_flags & 0x04)
        extra_kwargs_ok = bool(func_code.co_flags & 0x08)

        # -- assert that my understanding of calling protocol is correct
        #
        # param_names: the sequence of function parameter names
        # args_param: [optional] the name of the *vargs parameter
        # kwargs_param: [optional] the name of the **kwargs parameter
        # pos_params: sequence of potentially-positional parameter names
        try:
            if extra_args_ok and extra_kwargs_ok:
                assert len(bind_varnames) >= co_argcount + 2
                param_names = bind_varnames[:co_argcount + 2]
                args_param = param_names[co_argcount]
                kwargs_param = param_names[co_argcount + 1]
                pos_params = param_names[:co_argcount]
            elif extra_kwargs_ok:
                assert len(bind_varnames) >= co_argcount + 1
                param_names = bind_varnames[:co_argcount + 1]
                kwargs_param = param_names[co_argcount]
                pos_params = param_names[:co_argcount]
            elif extra_args_ok:
                assert len(bind_varnames) >= co_argcount + 1
                param_names = bind_varnames[:co_argcount + 1]
                args_param = param_names[co_argcount]
                pos_params = param_names[:co_argcount]
            else:
                assert len(bind_varnames) >= co_argcount
                param_names = bind_varnames[:co_argcount]
                pos_params = param_names[:co_argcount]
        except AssertionError:
            print 'YIKES: MISUNDERSTANDING OF CALL PROTOCOL:',
            print co_argcount,
            print bind_varnames,
            print '%x' % func_code.co_flags
            raise

        if len(args) > co_argcount and not extra_args_ok:
            raise TypeError('Argument count exceeds number of positional params')

        # -- bind positional arguments
        for i, (param_i, arg_i) in enumerate(zip(param_names, args)):
            assert bind_varnames[i] == param_i
            _locals[i + bind_offset] = arg_i

        if extra_args_ok:
            _locals[co_varnames.index(args_param)] == args[co_argcount:]

        # -- bind keyword arguments
        if extra_kwargs_ok:
            kwargs_pos = co_varnames.index(kwargs_param)
            _locals[kwargs_pos] == {}

        for aname, aval in kwargs.items():
            try:
                pos = pos_params.index(aname) + bind_offset
            except ValueError:
                if extra_kwargs_ok:
                    _locals[kwargs_pos][aname] = aval
                    continue
                else:
                    raise TypeError('Unrecognized keyword argument', aname)
            if _locals[pos] == Unassigned:
                _locals[pos] = aval
            else:
                raise TypeError('Duplicate argument for parameter', aname)

        # -- find default values
        if func.func_defaults:
            defaults = func.func_defaults
            for ii, val in enumerate(defaults):
                if _locals[co_argcount - len(defaults) + ii] is Unassigned:
                   _locals[co_argcount - len(defaults) + ii] = val

        # print 'BINDING'
        for name, lval in zip(co_varnames, _locals):
            if (isinstance(lval, np.ndarray)
                    and not id(lval) in self.watcher.svars):
                self.add_shadow(lval)
            #print '  locals:', name, lval, id(lval)

        self.code_iter = itercode(func_code.co_code)
        jmp = None
        while not hasattr(self, 'rval'):
            try:
                i, op, arg = self.code_iter.send(jmp)
            except StopIteration:
                break
            name = opcode.opname[op]
            name = {
                    'SLICE+0': 'SLICE_PLUS_0',
                    'SLICE+1': 'SLICE_PLUS_1',
                    'SLICE+2': 'SLICE_PLUS_2',
                    'SLICE+3': 'SLICE_PLUS_3',
                    }.get(name, name)
            if self.print_ops:
                print 'OP: ', i, name
            if self.print_stack:
                print self.stack
            jmp = getattr(self, 'op_' + name)(i, op, arg)

        return self.rval

    def op_BINARY_ADD(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        # No Theano vars allowed on the stack
        assert not hasattr(arg1, 'type')
        assert not hasattr(arg2, 'type')
        r = arg1 + arg2
        self.stack.append(r)
        if (id(arg1) in self.watcher.svars
                or id(arg2) in self.watcher.svars):
            s1 = self.ensure_shadow(arg1)
            s2 = self.ensure_shadow(arg2)
            if isinstance(r, np.ndarray):
                self.watcher.shadow(r, (s1 + s2).astype(str(r.dtype)))
            else:
                self.watcher.shadow(r, s1 + s2)
            #print 'added sym'

    def op_BINARY_DIVIDE(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        assert not hasattr(arg1, 'type')
        assert not hasattr(arg2, 'type')
        r = arg1 / arg2
        self.stack.append(r)
        if (id(arg1) in self.watcher.svars
                or id(arg2) in self.watcher.svars):
            s1 = self.ensure_shadow(arg1)
            s2 = self.ensure_shadow(arg2)
            if isinstance(r, np.ndarray):
                self.watcher.shadow(r, (s1 / s2).astype(str(r.dtype)))
            else:
                self.watcher.shadow(r, s1 / s2)

    def op_BINARY_FLOOR_DIVIDE(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        assert not hasattr(arg1, 'type')
        assert not hasattr(arg2, 'type')
        r = arg1 // arg2
        self.stack.append(r)
        if (id(arg1) in self.watcher.svars
                or id(arg2) in self.watcher.svars):
            s1 = self.ensure_shadow(arg1)
            s2 = self.ensure_shadow(arg2)
            if isinstance(r, np.ndarray):
                self.watcher.shadow(r, (s1 // s2).astype(str(r.dtype)))
            else:
                self.watcher.shadow(r, s1 // s2)

    def op_BINARY_SUBTRACT(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        assert not hasattr(arg1, 'type')
        assert not hasattr(arg2, 'type')
        r = arg1 - arg2
        self.stack.append(r)
        if (id(arg1) in self.watcher.svars
                or id(arg2) in self.watcher.svars):
            s1 = self.ensure_shadow(arg1)
            s2 = self.ensure_shadow(arg2)
            if isinstance(r, np.ndarray):
                self.watcher.shadow(r, (s1 - s2).astype(str(r.dtype)))
            else:
                self.watcher.shadow(r, s1 - s2)

    def op_BINARY_MULTIPLY(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        r = arg1 * arg2
        self.stack.append(r)
        assert not hasattr(arg1, 'type')
        assert not hasattr(arg2, 'type')
        if (id(arg1) in self.watcher.svars
                or id(arg2) in self.watcher.svars):
            s1 = self.ensure_shadow(arg1)
            s2 = self.ensure_shadow(arg2)
            if isinstance(r, np.ndarray):
                self.watcher.shadow(r, (s1 * s2).astype(str(r.dtype)))
            else:
                self.watcher.shadow(r, s1 * s2)
            #print 'mul sym', id(r)

    def op_BINARY_POWER(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        r = arg1 ** arg2
        self.stack.append(r)
        if (id(arg1) in self.watcher.svars
                or id(arg2) in self.watcher.svars):
            s1 = self.ensure_shadow(arg1)
            s2 = self.ensure_shadow(arg2)
            self.watcher.shadow(r, s1 ** s2)
            #print 'mul sym', id(r)

    def op_BINARY_SUBSCR(self, i, op, arg):
        # Implements TOS = TOS1[TOS].
        tos1, tos = self.stack[-2:]
        #print 'tos', tos
        #print 'tos1', tos1
        rval = tos1[tos]
        self.stack[-2:] = [rval]
        w = self.watcher
        if id(tos) in w.svars or id(tos1) in w.svars:
            if id(tos) in w.svars:
                s_tos = w.svars[id(tos)]
                s_tos1 = self.ensure_shadow(tos1)
                s_rval = s_tos1[s_tos]
            elif type(tos) == int:
                # don't make a symbol for this constant yet
                s_tos1 = self.ensure_shadow(tos1)
                s_rval = s_tos1[tos]
            elif type(tos) == slice:
                raise NotImplementedError('x[slice]')
            elif type(tos) == tuple:
                assert id(tos1) in w.svars
                s_tos1 = w.svars[id(tos1)]
                s_rval = s_tos1.__getitem__(tos)
            else:
                raise NotImplementedError()
            w.shadow(rval, s_rval)

    def op_BUILD_MAP(self, i, op, arg):
        self.stack.append({})

    def op_BUILD_SLICE(self, i, op, arg):
        if arg == 2:
            tos1, tos = self.stack[-2:]
            self.stack[-2:] = [slice(tos1, tos)]
        elif arg == 3:
            tos2, tos1, tos = self.stack[-3:]
            self.stack[-3:] = [slice(tos2, tos1, tos)]
        else:
            raise NotImplementedError()

    def op_BUILD_TUPLE(self, i, op, arg):
        if arg:
            t = tuple(self.stack[-arg:])
            self.stack[-arg:] = [t]
        else:
            self.stack.append(())

    def op_CALL_FUNCTION(self, i, op, arg):
        n_args = arg & 0xFF
        n_kwargs = (arg & 0xFF00) >> 8
        assert not (arg >> 16) # what would this stuff up here mean?
        kwargs = dict([(self.stack[-2 * ii], self.stack[-2 * ii + 1])
                for ii in range(n_kwargs, 0, -1)])
        args = [self.stack[-ii - 2 * n_kwargs] for ii in range(n_args, 0, -1)]
        # -- pop all args off the stack
        if arg:
            self.stack = self.stack[:- n_args - 2 * n_kwargs]
        # -- pop the function itself off the stack
        func = self.stack.pop(-1)

        #print dir(func)
        #print func.__self__
        all_args = args + kwargs.values()
        s_args = [self.watcher.svars.get(id(a), a) for a in args]
        s_kwargs = dict([(kw, self.watcher.svars.get(id(val), val))
            for kw, val in kwargs.items()])

        if hasattr(func, '__theano_op__'):
            # XXX: document that we are assuming func is pure -
            #      if rval depends on globals or closure this Context is not
            #      going to know that.
            # -- hand control back to Python for duration of func
            rval = func(*args, **kwargs)
            if any(id(a) in self.watcher.svars for a in all_args):
                s_rval = func.__theano_op__(*s_args, **s_kwargs)
                self.watcher.shadow(rval, s_rval)
        elif (
                (getattr(func, '__module__', None)
                    and func.__module__.startswith('numpy'))
                or isinstance(func, np.ufunc)
                or str(func) == '<built-in function abs>'
                or str(func) == '<built-in function max>'
                ):

            rval = func(*args, **kwargs)
            all_args = args + kwargs.values()
            if any(id(a) in self.watcher.svars for a in all_args):
                if kwargs:
                    # TODO put kwargs into the watcher calls
                    raise NotImplementedError()
                if 0: pass
                elif func.__name__ == 'abs':
                    self.watcher.shadow(rval, abs(*s_args))
                elif func.__name__ == 'any':
                    print 'WARNING: ignoring dependency through np.any'
                elif func.__name__ == 'dot':
                    self.watcher.shadow(rval, theano.tensor.dot(*s_args))
                elif func.__name__ == 'exp':
                    self.watcher.shadow(rval, theano.tensor.exp(*s_args))
                elif func.__name__ == 'log':
                    self.watcher.shadow(rval, theano.tensor.log(*s_args))
                elif func.__name__ == 'log10':
                    self.watcher.shadow(rval, theano.tensor.log10(*s_args))
                elif func.__name__ == 'maximum':
                    self.watcher.shadow(rval, theano.tensor.maximum(*s_args))
                elif func.__name__ == 'max':
                    assert str(func) == '<built-in function max>'
                    # N.B. builtin max -> tensor.maximum
                    s_rval = theano.tensor.maximum(*s_args)
                    assert s_rval.ndim == 0  # builtin max can't make vector
                    self.watcher.shadow(rval, s_rval)
                elif func.__name__ == 'mean':
                    self.watcher.shadow(rval, theano.tensor.mean(*s_args))
                elif func.__name__ == 'sum':
                    self.watcher.shadow(rval, theano.tensor.sum(*s_args))
                elif func.__name__ == 'zeros_like':
                    self.watcher.shadow(rval, theano.tensor.zeros_like(*s_args))
                else:
                    raise NotImplementedError(func)
            else:
                # no argument was shadowed (e.g. zeros())
                if isinstance(rval, np.ndarray):
                    self.add_shadow(rval)
        elif isinstance(getattr(func, '__self__', None), np.ndarray):
            assert id(func.__self__) in self.watcher.svars
            s_self = self.watcher.svars[id(func.__self__)]

            if 0: pass
            elif func.__name__ == 'copy':
                assert not args
                assert not kwargs
                rval = func()
                self.watcher.shadow(rval, s_self.copy())
            elif func.__name__ == 'mean':
                rval = func(*args, **kwargs)
                self.watcher.shadow(rval, s_self.mean(*s_args, **s_kwargs))
            elif func.__name__ == 'reshape':
                rval = func(*args, **kwargs)
                self.watcher.shadow(rval, s_self.reshape(*s_args, **s_kwargs))
            elif func.__name__ == 'sum':
                rval = func(*args, **kwargs)
                self.watcher.shadow(rval, s_self.sum(*s_args, **s_kwargs))
            else:
                raise NotImplementedError()
        elif 'built-in' in str(func):
            # -- built-in ndarray methods should be caught above, not here.
            if func.__name__ in ('setdefault', 'range'):
                rval = func(*args, **kwargs)
            else:
                raise NotImplementedError(func)
        else:
            logger.debug('stepping into %s' % str(func))
            vm = FrameVM(self.watcher, func)
            rval = vm.call(args, kwargs)
        self.stack.append(rval)

    def op_COMPARE_OP(self, i, op, arg):
        opname = opcode.cmp_op[arg]
        right = self.stack.pop(-1)
        left = self.stack.pop(-1)
        if 0: pass
        elif opname == '==': self.stack.append(left == right)
        elif opname == '!=': self.stack.append(left != right)
        elif opname == '>': self.stack.append(left > right)
        elif opname == '<': self.stack.append(left < right)
        elif opname == 'is': self.stack.append(left is right)
        else:
            raise NotImplementedError('comparison: %s' % opname)

        if any(id(a) in self.watcher.svars for a in [left, right]):
            sargs = [self.watcher.svars.get(id(a), a) for a in [left, right]]
            tos = self.stack[-1]
            if 0: pass
            elif opname == '<':
                self.watcher.shadow(tos, theano.tensor.lt(left, right))
            elif opname == '>':
                self.watcher.shadow(tos, theano.tensor.gt(left, right))
            else:
                raise NotImplementedError()

    def op_DUP_TOPX(self, i, op, arg):
        assert arg > 0
        self.stack.extend(self.stack[-arg:])

    def op_FOR_ITER(self, i, op, arg):
        # either push tos.next()
        # or pop tos and send (arg)
        tos = self.stack[-1]
        try:
            next = tos.next()
            # print 'next', next
            self.stack.append(next)
        except StopIteration:
            self.stack.pop(-1)
            return ('rel', arg)

    def op_INPLACE_ADD(self, i, op, arg):
        tos = self.stack.pop(-1)
        tos1 = self.stack.pop(-1)

        r = tos1
        r += tos
        self.stack.append(r)
        if (id(tos) in self.watcher.svars
                or id(tos1) in self.watcher.svars):
            s_tos = self.ensure_shadow(tos)
            s_tos1 = self.ensure_shadow(tos1)
            self.watcher.shadow(r, s_tos + s_tos1)

    def op_INPLACE_MULTIPLY(self, i, op, arg):
        tos = self.stack.pop(-1)
        tos1 = self.stack.pop(-1)

        r = tos1
        r *= tos
        self.stack.append(r)
        if (id(tos) in self.watcher.svars
                or id(tos1) in self.watcher.svars):
            s_tos = self.ensure_shadow(tos)
            s_tos1 = self.ensure_shadow(tos1)
            self.watcher.shadow(r, s_tos * s_tos1)

    def op_INPLACE_SUBTRACT(self, i, op, arg):
        tos = self.stack.pop(-1)
        tos1 = self.stack.pop(-1)

        r = tos1
        r -= tos
        self.stack.append(r)
        if (id(tos) in self.watcher.svars
                or id(tos1) in self.watcher.svars):
            s_tos = self.ensure_shadow(tos)
            s_tos1 = self.ensure_shadow(tos1)
            self.watcher.shadow(r, s_tos - s_tos1)

    def op_JUMP_ABSOLUTE(self, i, op, arg):
        # print 'sending', arg
        return ('abs', arg)

    def op_JUMP_IF_TRUE(self, i, op, arg):
        tos = self.stack[-1]
        if tos:
            return ('rel', arg)

    def op_GET_ITER(self, i, op, arg):
        # replace tos -> iter(tos)
        tos = self.stack[-1]
        if id(tos) in self.watcher.svars:
            raise NotImplementedError('iterator of watched value')
        self.stack[-1] = iter(tos)

    def op_LOAD_GLOBAL(self, i, op, arg):
        # print 'LOAD_GLOBAL', self.names[arg]
        tos = self._myglobals[self.func.func_code.co_names[arg]]
        self.stack.append(tos)
        if (isinstance(tos, np.ndarray)
                and id(tos) not in self.watcher.svars):
            self.add_shadow(self.stack[-1])

    def op_LOAD_ATTR(self, i, op, arg):
        # print 'LOAD_ATTR', self.names[arg]
        attr = self.func.func_code.co_names[arg]
        #
        # we would like to do
        #    self.stack[-1] = getattr(TOS, attr)
        #
        # *EXCEPT* if attr is a property, then it actually represents a
        # function call
        tos = self.stack.pop(-1)

        if isinstance(tos, np.ndarray):
            if id(tos) not in self.watcher.svars:
                raise NotImplementedError('how did this var get here?',
                        (id(tos), tos))

        if id(tos) in self.watcher.svars:
            s_tos = self.watcher.svars[id(tos)]

            # hard-code of how to deal with every ndarray property :/
            # XXX: think of how not to list all of the methods twice (!) as in
            # both here and in the CALL_FUNCTION handler
            if attr in ('copy', 'dtype', 'mean', 'reshape', 'sum'):
                rval = getattr(tos, attr)
            elif attr == 'shape':
                rval = tos.shape
                self.watcher.shadow(rval, s_tos.shape)
            elif attr == 'T':
                rval = tos.T
                self.watcher.shadow(rval, s_tos.T)
            else:
                raise NotImplementedError('ndarray attribute %s' % attr)
            self.stack.append(rval)
        else:
            logger.debug('attribute access %s' % attr)
            rval = getattr(tos, attr)
            self.stack.append(rval)
            if (isinstance(rval, np.ndarray)
                    and id(rval) not in self.watcher.svars):
                self.add_shadow(rval)

    def op_LOAD_CONST(self, i, op, arg):
        self.stack.append(self.func.func_code.co_consts[arg])
        tos = self.stack[-1]
        if (isinstance(tos, np.ndarray)
                and id(tos) not in self.watcher.svars):
            raise NotImplementedError()

    def op_LOAD_DEREF(self, i, op, arg):
        # print '???', i, op, arg
        # print self.func.func_closure
        thing = self.func.func_closure[arg]
        # print dir(thing.cell_contents)
        self.stack.append(thing.cell_contents)
        tos = self.stack[-1]
        if (isinstance(tos, np.ndarray)
                and id(tos) not in self.watcher.svars):
            self.add_shadow(tos)

    def op_LOAD_FAST(self, i, op, arg):
        #print 'LOAD_FAST', self.func.func_code.co_varnames[arg], self._locals[arg]
        tos = self._locals[arg]
        self.stack.append(tos)
        if (isinstance(tos, np.ndarray)
                and id(tos) not in self.watcher.svars):
            self.add_shadow(tos)

    def op_POP_BLOCK(self, i, op, arg):
        #print 'pop block, what to do?'
        pass

    def op_POP_JUMP_IF_FALSE(self, i, op, arg):
        #tos = self.stack[-1]
        tos = self.stack.pop(-1)
        if not tos:
            return ('abs', arg)

    def op_POP_JUMP_IF_TRUE(self, i, op, arg):
        #tos = self.stack[-1]
        tos = self.stack.pop(-1)
        if tos:
            return ('abs', arg)

    def op_POP_TOP(self, i, op, arg):
        self.stack.pop(-1)

    def op_PRINT_ITEM(self, i, op, arg):
        thing = self.stack.pop(-1)
        if str(thing) == 'PRINT_OPS:True':
            self.print_ops = True
        if str(thing) == 'PRINT_STACK:True':
            self.print_stack = True
        print thing,

    def op_PRINT_NEWLINE(self, i, op, arg):
        print ''

    def op_SETUP_LOOP(self, i, op, arg):
        #print 'SETUP_LOOP, what to do?'
        pass

    def op_SLICE_PLUS_3(self, i, op, arg):
        # Implements TOS = TOS2[TOS1:TOS]
        TOS2, TOS1, TOS = self.stack[-3:]
        rval = TOS2[TOS1:TOS]
        self.stack[-3:] = [rval]

        watcher = self.watcher
        if any(id(t) in watcher.svars for t in [TOS, TOS1, TOS2]):
            s  = w.get(TOS)
            s1 = w.get(TOS1)
            s2 = w.get(TOS2)
            s_rval = s2[s1:s]
            self.watcher.shadow(rval, s_rval)


    def op_STORE_FAST(self, i, op, arg):
        #print 'STORE_FAST', self.varnames[arg], self.stack[-1]
        self._locals[arg] = self.stack.pop(-1)

    def op_STORE_MAP(self, i, op, arg):
        key = self.stack.pop(-1)
        val = self.stack.pop(-1)
        dct = self.stack[-1]
        dct[key] = val

    def op_STORE_SUBSCR(self, i, op, arg):
        # Implements TOS1[TOS] = TOS2.
        tos = self.stack.pop(-1)
        tos1 = self.stack.pop(-1)
        tos2 = self.stack.pop(-1)

        tos1[tos] = tos2

        watcher = self.watcher
        svars = watcher.svars
        # tos can't be real-valued so there's no gradient through it
        if id(tos1) in svars or id(tos2) in svars:
            s_tos1 = self.ensure_shadow(tos1)
            s_tos2 = self.ensure_shadow(tos2)

            new_s_tos1 = theano.tensor.set_subtensor(s_tos1[tos], s_tos2)
            svars[id(tos1)] = new_s_tos1


    def op_RAISE_VARARGS(self, i, op, arg):
        if 1 <= arg:
            exc = self.stack.pop(-1)
        if 2 <= arg:
            param = self.stack.pop(-1)
        if 3 <= arg:
            tb = self.stack.pop(-1)
        raise NotImplementedError('exception handling')

    def op_RETURN_VALUE(self, i, op, arg):
        self.rval = self.stack.pop(-1)

    def op_ROT_TWO(self, i, op, arg):
        a = self.stack[-1]
        b = self.stack[-2]
        self.stack[-1] = b
        self.stack[-2] = a

    def op_ROT_THREE(self, i, op, arg):
        a = self.stack[-1]
        b = self.stack[-2]
        c = self.stack[-3]
        self.stack[-1] = b
        self.stack[-2] = c
        self.stack[-3] = a

    def op_UNPACK_SEQUENCE(self, i, op, arg):
        tos = self.stack.pop(-1)
        self.stack.extend(tos[::-1])


class Context(object):
    def __init__(self):
        self.svars = {}
        self.nogc = [] # ids that must not be reused
        # XXX: rethink to avoid actually holding on to all these intermediates.

    def shadow(self, rval, sval):
        assert hasattr(sval, 'type')  # assert sval is Theano variable
        self.svars.setdefault(id(rval), sval)
        # -- shadow vars have to match dtype and ndim
        if isinstance(rval, np.ndarray):
            assert str(rval.dtype) == sval.dtype, (rval, sval)
            assert rval.ndim == sval.ndim, (rval, sval)
        # -- assert postcondition
        assert sval is self.svars[id(rval)]
        self.nogc.append(rval)

    def call(self, fn, args=(), kwargs={}):
        vm = FrameVM(self, fn)
        return vm.call(args, kwargs)

