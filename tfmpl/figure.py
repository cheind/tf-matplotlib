# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

import tensorflow as tf
import traceback
import numpy as np
from functools import wraps

from tfmpl.meta import vararg_decorator, as_list
from tfmpl.meta import PositionalTensorArgs

def figure_buffer(figs):
    '''Extract raw image buffer from matplotlib figure shaped as 1xHxWx3.'''  
    buffers = []
    w, h = figs[0].canvas.get_width_height()
    for f in figs:
        wf, hf = f.canvas.get_width_height()
        assert wf == w and hf == h, 'All canvas objects need to have same size'
        buffers.append(np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3))

    return np.stack(buffers) # NxHxWx3

@vararg_decorator
def figure_tensor(func, **tf_pyfunc_kwargs):
    '''Decorate matplotlib drawing routines.

    This dectorator is meant to decorate functions that return matplotlib
    figures. The decorated function has to have the following signature

        def decorated(*args, **kwargs) -> figure or iterable of figures
    
    where `*args` can be any positional argument and `**kwargs` are any
    keyword arguments. The decorated function returns a tensor of shape 
    `[NumFigures, Height, Width, 3]` of type `tf.uint8`. 
    
    The drawing code is invoked during running of TensorFlow sessions, 
    at a time when all positional tensor arguments have been evaluated 
    by the session. The decorated function is then passed the tensor values. 
    All non tensor arguments remain unchanged.
    '''

    name = tf_pyfunc_kwargs.pop('name', func.__name__)

    @wraps(func)
    def wrapper(*func_args, **func_kwargs):
        tf_args = PositionalTensorArgs(func_args)
        
        def pyfnc_callee(*tensor_values, **unused):
            try:
                figs = as_list(func(*tf_args.mix_args(tensor_values), **func_kwargs))
                for f in figs:
                    f.canvas.draw()
                return figure_buffer(figs)
            except Exception:
                print('-'*5 + 'tfmpl catched exception' + '-'*5)
                print(traceback.format_exc())                
                print('-'*20)
                raise

        return tf.py_func(pyfnc_callee, tf_args.tensor_args, tf.uint8, name=name, **tf_pyfunc_kwargs)
    return wrapper

@vararg_decorator
def blittable_figure_tensor(func, init_func, **tf_pyfunc_kwargs):
    '''Decorate matplotlib drawing routines with blitting support.

    This dectorator is meant to decorate functions that return matplotlib
    figures. The decorated function has to have the following signature

        def decorated(*args, **kwargs) -> iterable of artists
    
    where `*args` can be any positional argument and `**kwargs` are any
    keyword arguments. The decorated function returns a tensor of shape 
    `[NumFigures, Height, Width, 3]` of type `tf.uint8`. 

    Besides the actual drawing function, `blittable_figure_tensor` requires
    a `init_func` argument with the following signature

        def init(*args, **kwargs) -> iterable of figures, iterable of artists
    
    The init function is meant to create and initialize figures, as well as to
    perform drawing that is meant to be done only once. Any set of artits to be
    updated in later drawing calls should also be allocated in init. The 
    initialize function must have the same positional and keyword arguments
    as the decorated function. It is called once before the decorated function
    is called.
    
    The drawing code / init function is invoked during running of TensorFlow 
    sessions, at a time when all positional tensor arguments have been 
    evaluated by the session. The decorated / init function is then passed the 
    tensor values. All non tensor arguments remain unchanged.
    '''
    name = tf_pyfunc_kwargs.pop('name', func.__name__)
    assert callable(init_func), 'Init function not callable'

    @wraps(func)
    def wrapper(*func_args, **func_kwargs):
        figs = None
        bgs = None

        tf_args = PositionalTensorArgs(func_args)

        def pyfnc_callee(*tensor_values, **unused):
            
            try:
                nonlocal figs, bgs
                pos_args = tf_args.mix_args(tensor_values)
                
                if figs is None:
                    figs, artists = init_func(*pos_args, **func_kwargs)
                    figs = as_list(figs)
                    artists = as_list(artists)
                    for f in figs:
                        f.canvas.draw()
                    for a in artists:
                        a.set_animated(True)
                    bgs = [f.canvas.copy_from_bbox(f.bbox) for f in figs]                

                artists = as_list(func(*pos_args, **func_kwargs))

                for f, bg in zip(figs, bgs):
                    f.canvas.restore_region(bg)                
                for a in artists:
                    a.axes.draw_artist(a)
                for f in figs:
                    f.canvas.blit(f.bbox)

                return figure_buffer(figs)
            except Exception:
                print('-'*5 + 'tfmpl catched exception' + '-'*5)
                print(traceback.format_exc())                
                print('-'*20)
                raise

        return tf.py_func(pyfnc_callee, tf_args.tensor_args, tf.uint8, name=name, **tf_pyfunc_kwargs)
    return wrapper