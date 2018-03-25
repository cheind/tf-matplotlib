import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from datetime import datetime
now = datetime.now()
import tfmpl
import os

if __name__ == '__main__':

    
    with tf.Session(graph=tf.Graph()) as sess:

        def beale(x, y): 
            return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
        

        def create_drawer():
            
            fig = None
            ax = None
            ln = None
            xyzs = [[], [], []]

            def init_fig(xy):
                print('in init')
                nonlocal fig, ax, ln
                fig = tfmpl.create_figure()
                ax = fig.add_subplot(111, projection='3d', elev=50, azim=-50)

                xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, 40), np.linspace(-4.5, 4.5, 40))
                zz = beale(xx, yy)
                ax.plot_surface(xx, yy, zz, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=plt.cm.jet)
                ax.plot([3], [.5], [beale(3, .5)], 'k*', markersize=10)

                ln, = ax.plot([],[],[] ,animated=True)

                return fig, None
                
            @tfmpl.blittable_figure_tensor(init_func=init_fig)
            def draw(xy):
                xyzs[0].append(xy[0, 0])
                xyzs[1].append(xy[0, 1])
                xyzs[2].append(beale(xy[0, 0], xy[0, 1]))

                ln.set_data(xyzs[0], xyzs[1])
                ln.set_3d_properties(xyzs[2])
                return ln,

            return draw
        
        xy = tf.get_variable('xy', [1, 2], tf.float32, tf.constant_initializer([[3, 4.]]))
        z = beale(xy[:, 0], xy[:, 1])

        opt = tf.train.AdagradOptimizer(1e-2)        
        grads_and_vars = opt.compute_gradients(z[0], [xy])
        clipped_grads_and_vars = [(tf.clip_by_value(g, -10, 10), v) for g, v in grads_and_vars]
        train = opt.apply_gradients(clipped_grads_and_vars)

        image_tensor = create_drawer()(xy)
        image_summary = tf.summary.image('sgd', image_tensor)        
        all_summaries = tf.summary.merge_all()

        os.makedirs('log', exist_ok=True)
        logdir = "log/" + now.strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(2000):              
            if i % 25 == 0:
                summary = sess.run(all_summaries)
                writer.add_summary(summary, global_step=i)
                writer.flush()
            sess.run(train)