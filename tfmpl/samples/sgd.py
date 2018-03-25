import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import tfmpl
import os

if __name__ == '__main__':

    
    with tf.Session(graph=tf.Graph()) as sess:

        def beale(x, y): 
            return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
        

        def create_drawer():
            
            fig = None
            ax = None
            x, y, z = [], [], []

            def init_fig(xy):
                nonlocal fig, ax
                fig = tfmpl.create_figure()
                ax = fig.add_subplot(111, projection='3d', elev=50, azim=-50)

                xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, 40), np.linspace(-4.5, 4.5, 40))
                zz = beale(xx, yy)
                ax.plot_surface(xx, yy, zz, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=plt.cm.jet)
                ax.plot([3], [.5], [beale(3, .5)], 'k*', markersize=10)

                return fig, None
                
            @tfmpl.blittable_figure_tensor(init_func=init_fig)
            def draw(xy):
                pass

            print(draw)
            return draw

        """
        @tfmpl.figure_summary(name='sgd')
        def draw(x, y):   
            fig = tfmpl.create_figure()
            ax = fig.add_subplot(111, projection='3d', elev=50, azim=-50)

            xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, 40), np.linspace(-4.5, 4.5, 40))
            zz = beale(xx, yy)
            ax.plot_surface(xx, yy, zz, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=plt.cm.jet)
            ax.plot([3], [.5], [beale(3, .5)], 'k*', markersize=10)

            #ax.plot(x, y, beale(x,y), 'o')

            return fig
        """
        
        xy = tf.get_variable('xy', [2, 2], tf.float32, tf.constant_initializer(3))
        z = beale(xy[:, 0], xy[:, 1])

        opt = tf.train.GradientDescentOptimizer(1e-6)
        train = opt.minimize(z[0])

        image_tensor = create_drawer()(xy)
        image_summary = tf.summary.image('sgd', image_tensor)        
        all_summaries = tf.summary.merge_all()

        os.makedirs('log', exist_ok=True)
        writer = tf.summary.FileWriter('./log', sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(100):  
            summary, _ = sess.run([all_summaries, train])
            #summary, _ = sess.run([image_tensor, train])
            writer.add_summary(summary, global_step=i)

        
        """
        scale = tf.placeholder(tf.float32)        
        scaled = points*scale

        #img = draw_scatter(scaled, color='b')
        #imgdata = sess.run(img, feed_dict={scale: 2.})
        #plt.imshow(imgdata)        
        #plt.show()

        image_summary = cached_draw(scaled)
        print(image_summary)
        all_summaries = tf.summary.merge_all()

        os.makedirs('log', exist_ok=True)
        writer = tf.summary.FileWriter('./log', sess.graph)

        for i in range(10):
            summary = sess.run(all_summaries, feed_dict={scale: 2.})
            writer.add_summary(summary, global_step=i)



    

    xmin, xmax, xstep = -4.5, 4.5, .2
    ymin, ymax, ystep = -4.5, 4.5, .2

    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    z = beale(x, y)

    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes(projection='3d', elev=50, azim=-50)

    ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=plt.cm.jet)
    ax.plot([3], [.5], [beale(3, .5)], 'k*', markersize=10)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    plt.show()
    """