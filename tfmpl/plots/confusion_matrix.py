# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

import numpy as np
import re
from textwrap import wrap
from itertools import product

def from_labels_and_predictions(labels, predictions, num_classes):
    '''Compute a confusion matrix from labels and predictions.
    
    A drop-in replacement for tf.confusion_matrix that works on CPU data
    and not tensors.

    Params
    ------
    labels : array-like 
        1-D array of real labels for classification
    predicitions: array-like
        1-D array of predicted label classes
    num_classes: scalar
        Total number of classes

    Returns
    -------
    matrix : NxN array
        Array of shape [num_classes, num_classes] containing the confusion values.
    ''' 
    assert len(labels) == len(predictions)   
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for i in range(len(labels)):
        cm[labels[i], predictions[i]] += 1
    return cm

def draw(ax, cm, axis_labels=None, normalize=False):
    '''Plot a confusion matrix.

    Inspired by
    https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard

    Params
    ------
    ax : axis
        Axis to plot on
    cm : NxN array
        Confusion matrix
    
    Kwargs
    ------
    axis_labels : array-like
        Array of size N containing axis labels
    normalize : bool
        Whether to plot counts or ratios.
    '''
    
    cm = np.asarray(cm)
    num_classes = cm.shape[0]

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