# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

import numpy as np
import re
from textwrap import wrap
from itertools import product

def confusion_matrix(ax, labels, predictions, num_classes, axis_labels=None, normalize=False):
    '''Render a confusion matrix.

    Inspired by
    https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard

    Note, training is tuned to give rather slow convergence, so that the change in confusion matrix
    is easier to spot. 
    '''
    
    assert len(labels) == len(predictions)

    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for i in range(len(labels)):
        cm[labels[i], predictions[i]] += 1

    if normalize:
        with np.errstate(invalid='ignore', divide='ignore'):
            cm = cm / cm.sum(1, keepdims=True)
        cm = np.nan_to_num(cm, copy=True)

    po = np.get_printoptions()
    np.set_printoptions(precision=2)
    
    ax.imshow(cm, cmap='Oranges')

    ticks = np.arange(num_classes)
    
    ax.set_xlabel('Predicted')
    ax.set_xticks(ticks)
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('Actual')
    ax.set_yticks(ticks)
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    if axis_labels is not None:
        ticklabels = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in axis_labels]
        ticklabels = ['\n'.join(wrap(l, 20)) for l in ticklabels]
        ax.set_xticklabels(ticklabels, rotation=-90,  ha='center')
        ax.set_yticklabels(ticklabels, va ='center')

    for i, j in product(range(num_classes), range(num_classes)):
        if cm[i,j] == 0:
            txt = '.'
        elif normalize:            
            txt = '{:.2f}'.format(cm[i,j])
        else:
            txt = '{}'.format(cm[i,j])
        ax.text(j, i, txt, horizontalalignment="center", verticalalignment='center', color= "black", fontsize=7)

    np.set_printoptions(**po)