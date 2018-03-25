import tensorflow as tf
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
    name = tf_pyfunc_kwargs.pop('name', func.__name__)

    @wraps(func)
    def wrapper(*func_args, **func_kwargs):

        # Args might be a mix of tensors and non-tensors.
        # We split them here and pass all tensorflow tensors
        # as inputs to py_func. This ensures that those
        # tensors will get evaluated before draw() is called
        # and tensor values will be provided.
        tf_args = PositionalTensorArgs(func_args)
        
        def pyfnc_callee(*tensor_values, **unused):
            figs = as_list(func(*tf_args.mix_args(tensor_values), **func_kwargs))
            for f in figs:
                f.canvas.draw()
            return figure_buffer(figs)

        return tf.py_func(pyfnc_callee, tf_args.tensor_args, tf.uint8, name=name, **tf_pyfunc_kwargs)
    return wrapper

@vararg_decorator
def blittable_figure_tensor(func, init_func, **tf_pyfunc_kwargs):
    name = tf_pyfunc_kwargs.pop('name', func.__name__)
    assert callable(init_func), 'Init function not callable'

    @wraps(func)
    def wrapper(*func_args, **func_kwargs):
        figs = None
        bgs = None

        tf_args = PositionalTensorArgs(func_args)

        def pyfnc_callee(*tensor_values, **unused):
            pos_args = tf_args.mix_args(tensor_values)
            nonlocal figs, bgs
            if figs is None and init_func:
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

        return tf.py_func(pyfnc_callee, tf_args.tensor_args, tf.uint8, name=name, **tf_pyfunc_kwargs)
    return wrapper