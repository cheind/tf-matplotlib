from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def create_figure(*fig_args, **fig_kwargs):
    fig = Figure(*fig_args, **fig_kwargs)
    # Attach canvas
    FigureCanvas(fig)
    return fig