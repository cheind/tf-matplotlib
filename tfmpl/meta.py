# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

from functools import wraps
import tensorflow as tf
from tensorflow.contrib.framework import is_tensor
from collections import Sequence

def vararg_decorator(f):
    '''Decorator to handle variable argument decorators.'''

    @wraps(f)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return f(args[0])
        else:
            return lambda realf: f(realf, *args, **kwargs)

    return decorator

class PositionalTensorArgs:
    '''Handle tensor arguments.'''

    def __init__(self, args):
        self.args = args
        self.tf_args = [(i,a) for i,a in enumerate(args) if is_tensor(a)]

    @property
    def tensor_args(self):
        return [a for i,a in self.tf_args]

    def mix_args(self, tensor_values):
        args = list(self.args)
        for i, (j, _) in enumerate(self.tf_args):
            args[j] = tensor_values[i]
        return args

def as_list(x):
    '''Ensure `x` is of list type.'''
    
    if x is None:
        x = []
    elif not isinstance(x, Sequence):
        x = [x]
    return list(x)