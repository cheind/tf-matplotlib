import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tfmpl
import os

if __name__ == '__main__':

    with tf.Session(graph=tf.Graph()) as sess:
        
        def cached_draw(scaled):
            figs = None

            @tfmpl.figure_summary(name='animation')
            def draw(scaled):   
                nonlocal figs

                if figs is None:
                    figs = tfmpl.create_figures(1, figsize=(4,4))
                    ax = figs[0].add_subplot(111)
                    ax.axis('off')
                    ax.set_xlim(-5,5)
                    ax.set_ylim(-5,5)
                    figs[0].tight_layout()
                else:
                    ax = figs[0].get_axes()[0]
            
                x = np.random.uniform(-5,5, size=(1,2))            
                ax.scatter(x[:, 0], x[:, 1], c='g')
                return figs

            return draw(scaled)


        points = tf.constant(
            np.random.normal(loc=0.0, scale=1.0, size=(100, 2)).astype(np.float32)
        )

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

