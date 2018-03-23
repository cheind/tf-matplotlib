from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def create_figure(*fig_args, **fig_kwargs):
    fig = Figure(*fig_args, **fig_kwargs)
    # Attach canvas
    FigureCanvas(fig)
    return fig

def create_figures(n, *fig_args, **fig_kwargs):
    return [create_figure(*fig_args, **fig_kwargs) for _ in range(n)]