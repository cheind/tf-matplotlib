import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tfmpl
import os

if __name__ == '__main__':

    with tf.Session(graph=tf.Graph()) as sess:
        
        @tfmpl.figure_summary(name='x')
        def draw_scatter(scaled, color='b'):     
            fig = tfmpl.create_figure(figsize=(4,4))
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.scatter(scaled[:, 0], scaled[:, 1], c=color)
            fig.tight_layout()
            return fig  

        points = tf.constant(
            np.random.normal(loc=0.0, scale=1.0, size=(100, 2)).astype(np.float32)
        )

        scale = tf.placeholder(tf.float32)        
        scaled = points*scale

        #img = draw_scatter(scaled, color='b')
        #imgdata = sess.run(img, feed_dict={scale: 2.})
        #plt.imshow(imgdata)        
        #plt.show()

        image_summary = draw_scatter(scaled)
        print(image_summary)
        all_summaries = tf.summary.merge_all()

        os.makedirs('log', exist_ok=True)
        writer = tf.summary.FileWriter('./log', sess.graph)

        summary = sess.run(all_summaries, feed_dict={scale: 2.})
        writer.add_summary(summary, global_step=0)
