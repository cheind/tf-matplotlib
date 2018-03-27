# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import cm
from datetime import datetime
import tensorflow as tf
import numpy as np
import os

import tfmpl

if __name__ == '__main__':

    with tf.Session(graph=tf.Graph()) as sess:

        def beale(x, y):
            '''Beale surface for optimization tests.'''
            with tf.name_scope('beale', [x, y]):
                return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

        # List of optimizers to compare
        optimizers = [
            (tf.train.GradientDescentOptimizer(1e-3), 'SGD'),
            (tf.train.AdagradOptimizer(1e-1), 'Adagrad'),
            (tf.train.AdadeltaOptimizer(1e2), 'Adadelta'),
            (tf.train.AdamOptimizer(1e-1), 'Adam'),            
        ]

        paths = []        
        history = []

        def init_fig(*args, **kwargs):
            '''Initialize figures.'''
            fig = tfmpl.create_figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection='3d', elev=50, azim=-30)
            ax.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0))
            ax.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0))
            ax.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0))
            ax.set_title('Gradient descent on Beale surface')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_zlabel('beale($x$,$y$)')
        
            xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, 40), np.linspace(-4.5, 4.5, 40))
            zz = beale(xx, yy)
            ax.plot_surface(xx, yy, zz, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=cm.jet)
            ax.plot([3], [.5], [beale(3, .5)], 'k*', markersize=5)
            
            for o in optimizers:
                path, = ax.plot([],[],[], label=o[1])
                paths.append(path)

            ax.legend(loc='upper left')
            fig.tight_layout()

            return fig, paths
            
        @tfmpl.blittable_figure_tensor(init_func=init_fig)
        def draw(xy, z):
            '''Updates paths for each optimizer.'''
            history.append(np.c_[xy, z])
            xyz = np.stack(history) #NxMx3
            for idx, path in enumerate(paths):
                path.set_data(xyz[:, idx, 0], xyz[:, idx, 1])
                path.set_3d_properties(xyz[:, idx, 2])

            return paths

        # Create variables for each optimizer
        start = tf.constant_initializer([3., 4.], dtype=tf.float32)
        xys = [tf.get_variable(f'xy_{o[1]}', 2, tf.float32, initializer=start) for o in optimizers]        
        zs = [beale(xy[0], xy[1]) for xy in xys]

        # Define optimization target
        train = []
        for idx, (opt, name) in enumerate(optimizers):
            grads_and_vars = opt.compute_gradients(zs[idx], xys[idx])
            clipped = [(tf.clip_by_value(g, -10, 10), v) for g, v in grads_and_vars]
            train.append(opt.apply_gradients(clipped))

        # Generate summary
        image_tensor = draw(tf.stack(xys), tf.stack(zs))
        image_summary = tf.summary.image('optimization', image_tensor)        
        all_summaries = tf.summary.merge_all()

        # Alloc summary writer
        os.makedirs('log', exist_ok=True)
        now = datetime.now()
        logdir = "log/" + now.strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, sess.graph)

        # Run optimization, write summary every now and then.
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(200):              
            if i % 10 == 0:
                summary = sess.run(all_summaries)
                writer.add_summary(summary, global_step=i)
                writer.flush()
            sess.run(train)