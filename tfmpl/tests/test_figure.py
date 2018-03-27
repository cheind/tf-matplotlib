# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

import tensorflow as tf
import tfmpl
import numpy as np

def test_arguments():

    debug = {}

    @tfmpl.figure_tensor
    def draw(a, b, c, d=None, e=None):
        debug['a'] = a
        debug['b'] = b
        debug['c'] = c
        debug['d'] = d
        debug['e'] = e

        return tfmpl.create_figure()

    with tf.Session(graph=tf.Graph()) as sess:
        a = tf.constant(0)
        c = tf.placeholder(tf.float32)

        tensor = draw(a, [0,1], c, d='d', e='e')
        sess.run(tensor, feed_dict={c: np.zeros((2,2))})

    assert debug['a'] == 0
    assert debug['b'] == [0,1]
    np.testing.assert_allclose(debug['c'], np.zeros((2,2)))
    debug['d'] = 'd'
    debug['e'] = 'e'


def test_arguments_blittable():

    debug = {}

    def init(a, b, c, d=None, e=None):
        debug['init_args'] = [a, b, c, d, e]
        return tfmpl.create_figure(), None
        
    @tfmpl.blittable_figure_tensor(init_func=init)
    def draw(a, b, c, d=None, e=None):
        debug['args'] = [a, b, c, d, e]

    with tf.Session(graph=tf.Graph()) as sess:
        a = tf.constant(0)
        c = tf.placeholder(tf.float32)

        tensor = draw(a, [0,1], c, d='d', e='e')
        sess.run(tensor, feed_dict={c: np.zeros((2,2))})

    assert debug['init_args'][0] == 0
    assert debug['init_args'][1] == [0,1]
    np.testing.assert_allclose(debug['init_args'][2], np.zeros((2,2)))
    assert debug['init_args'][3] == 'd'
    assert debug['init_args'][4] == 'e'

    assert debug['args'][0] == 0
    assert debug['args'][1] == [0,1]
    np.testing.assert_allclose(debug['args'][2], np.zeros((2,2)))
    assert debug['args'][3] == 'd'
    assert debug['args'][4] == 'e'

def test_callcount():

    debug = {}
    debug['called'] = 0
    debug['a'] = []

    @tfmpl.figure_tensor
    def draw(a):
        debug['called'] += 1
        debug['a'].append(a)        
        return tfmpl.create_figure()

    with tf.Session(graph=tf.Graph()) as sess:
        a = tf.placeholder(tf.float32)

        tensor = draw(a)

        for i in range(5):
            sess.run(tensor, feed_dict={a: i})

    assert debug['called'] == 5
    np.testing.assert_allclose(debug['a'], [0,1,2,3,4])

def test_callcount_blittable():
    
    debug = {}
    debug['init_called'] = 0
    debug['draw_called'] = 0
    debug['a'] = []
    debug['a_init'] = []

    def init(a):
        debug['init_called'] += 1
        debug['a_init'] = a
        return tfmpl.create_figure(), None

    @tfmpl.blittable_figure_tensor(init_func=init)
    def draw(a):
        debug['draw_called'] += 1
        debug['a'].append(a)        
        
    with tf.Session(graph=tf.Graph()) as sess:
        a = tf.placeholder(tf.float32)

        tensor = draw(a)

        for i in range(5):
            sess.run(tensor, feed_dict={a: i})

    assert debug['init_called'] == 1
    assert debug['draw_called'] == 5
    assert debug['a_init'] == 0
    np.testing.assert_allclose(debug['a'], [0,1,2,3,4])

def test_callcount_blittable():
    
    debug = {}
    debug['init_called'] = 0
    debug['draw_called'] = 0
    debug['a'] = []
    debug['a_init'] = []

    def init(a):
        debug['init_called'] += 1
        debug['a_init'] = a
        return tfmpl.create_figure(), None

    @tfmpl.blittable_figure_tensor(init_func=init)
    def draw(a):
        debug['draw_called'] += 1
        debug['a'].append(a)        
        
    with tf.Session(graph=tf.Graph()) as sess:
        a = tf.placeholder(tf.float32)

        tensor = draw(a)

        for i in range(5):
            sess.run(tensor, feed_dict={a: i})

    assert debug['init_called'] == 1
    assert debug['draw_called'] == 5
    assert debug['a_init'] == 0
    np.testing.assert_allclose(debug['a'], [0,1,2,3,4])

def test_draw():

    @tfmpl.figure_tensor
    def draw():
        figs = tfmpl.create_figures(2, figsize=(4,3), dpi=100)

        figs[0].patch.set_facecolor('red')
        figs[1].patch.set_facecolor((0, 1, 0))
        
        return figs

    with tf.Session(graph=tf.Graph()) as sess:
        a = tf.placeholder(tf.float32)

        tensor = draw()

        imgs = sess.run(tensor)
        assert imgs.shape == (2, 300, 400, 3)
        np.testing.assert_allclose(imgs[0], np.tile([255, 0, 0], (300, 400, 1)))
        np.testing.assert_allclose(imgs[1], np.tile([0, 255, 0], (300, 400, 1)))
            
def test_draw_blittable():
    import matplotlib.patches as patches

    rect = None

    def init(t):
        nonlocal rect
        fig = tfmpl.create_figure(figsize=(4,4), dpi=100)
        ax = fig.add_axes([0,0,1,1])
        ax.invert_yaxis()
        rect = ax.add_patch(patches.Rectangle((0,0), 0.1, 0.1, facecolor=(0,1,0)))
        return fig, rect
        
    @tfmpl.blittable_figure_tensor(init_func=init)
    def draw(t):
        rect.set_xy((t,t))
        return rect

    with tf.Session(graph=tf.Graph()) as sess:
        t = tf.placeholder(tf.float32)
        tensor = draw(t)

        imgs = sess.run(tensor, feed_dict={t:0})
        assert imgs.shape == (1, 400, 400, 3)

        np.testing.assert_allclose(imgs[0, :40, :40], np.tile([0, 255, 0], (40, 40, 1)))
        
        imgs = sess.run(tensor, feed_dict={t:0.5})
        assert imgs.shape == (1, 400, 400, 3)
        np.testing.assert_allclose(imgs[0, 200:240, 200:240], np.tile([0, 255, 0], (40, 40, 1)))
            