import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tfmpl
import os

if __name__ == '__main__':

    with tf.Session(graph=tf.Graph()) as sess:
        
        def init_figs():
            return 

        @tfmpl.figure_tensor(name='x')
        def draw_scatter(scaled, colors, figs=None):   
            figs = tfmpl.create_figures(2, figsize=(4,4))
            for idx, f in enumerate(figs):
                ax = f.add_subplot(111)
                ax.axis('off')
                ax.scatter(scaled[:, 0], scaled[:, 1], c=colors[idx])
                f.tight_layout()

            return figs  

        points = tf.constant(
            np.random.normal(loc=0.0, scale=1.0, size=(100, 2)).astype(np.float32)
        )

        scale = tf.placeholder(tf.float32)        
        scaled = points*scale

        #img = draw_scatter(scaled, color='b')
        #imgdata = sess.run(img, feed_dict={scale: 2.})
        #plt.imshow(imgdata)        
        #plt.show()

        image_tensor = draw_scatter(scaled, ['r', 'b'])
        image_summary = tf.summary.image('image', image_tensor)
        
        all_summaries = tf.summary.merge_all()

        img = sess.run(image_tensor, feed_dict={scale: 2.})
        plt.imshow(img[0])
        plt.show()


        """
        os.makedirs('log', exist_ok=True)
        writer = tf.summary.FileWriter('./log', sess.graph)

        summary = sess.run(all_summaries, feed_dict={scale: 2.})
        writer.add_summary(summary, global_step=0)
        """