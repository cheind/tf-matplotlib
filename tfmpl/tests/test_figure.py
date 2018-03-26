
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

    

def test_figure():
    assert True == True