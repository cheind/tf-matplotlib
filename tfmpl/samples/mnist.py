# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================
"""Show usage of confusion matrix visualization.

Using a simple MNIST classifier taken from
https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_softmax.py
"""

from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import tensorflow as tf
import numpy as np
import os

import tfmpl

@tfmpl.figure_tensor
def draw_confusion_matrix(labels, predictions):
    '''Draw confusion matrix for MNIST.'''
    fig = tfmpl.create_figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    ax.set_title('Confusion matrix for MNIST classification')
    
    tfmpl.draw.confusion_matrix(
        ax, 
        labels, predictions, 
        10, axis_labels=['Digit ' + str(x) for x in range(10)], 
        normalize=True)

    return fig

if __name__ == '__main__':    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    with tf.Session(graph=tf.Graph()) as sess:

        # Create the model
        x = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.matmul(x, W) + b

        y_ = tf.placeholder(tf.float32, [None, 10])
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
        )
        train = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)

        preds = tf.argmax(y, 1)
        labels = tf.argmax(y_, 1)

        image_tensor = draw_confusion_matrix(preds, labels)
        image_summary = tf.summary.image('confusion_matrix', image_tensor)
        all_summaries = tf.summary.merge_all()

        os.makedirs('log', exist_ok=True)
        now = datetime.now()
        logdir = "log/" + now.strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, sess.graph)
        
        # Train
        sess.run(tf.global_variables_initializer())
        for i in range(1000):            
            batch_xs, batch_ys = mnist.train.next_batch(10)
            sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

            if i % 10 == 0:
                summary = sess.run(all_summaries, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                writer.add_summary(summary, global_step=i)
                writer.flush()

        correct_prediction = tf.equal(preds, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    