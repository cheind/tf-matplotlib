# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def create_figure(*fig_args, **fig_kwargs):
    '''Create a single figure.

    Args and Kwargs are passed to `matplotlib.figure.Figure`.
    
    This routine is provided in order to avoid usage of pyplot which
    is stateful and not thread safe. As drawing routines in tf-matplotlib
    are called from py-funcs in their respective thread, avoid usage
    of pyplot where possible.
    '''

    fig = Figure(*fig_args, **fig_kwargs)
    # Attach canvas
    FigureCanvas(fig)
    return fig

def create_figures(n, *fig_args, **fig_kwargs):
    '''Create multiple figures.

    Args and Kwargs are passed to `matplotlib.figure.Figure`.
    
    This routine is provided in order to avoid usage of pyplot which
    is stateful and not thread safe. As drawing routines in tf-matplotlib
    are called from py-funcs in their respective thread, avoid usage
    of pyplot where possible.
    '''
    return [create_figure(*fig_args, **fig_kwargs) for _ in range(n)]