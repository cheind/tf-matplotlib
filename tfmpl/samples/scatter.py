import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tfmpl


if __name__ == '__main__':

    with tf.Session(graph=tf.Graph()) as sess:

        points = tf.constant(
            np.random.normal(loc=0.0, scale=1.0, size=(100, 2)).astype(np.float32)
        )

        scale = tf.placeholder(tf.float32)        
        scaled = points*scale

        @tfmpl.figure_summary(name='myscatterplot')
        def draw_scatter(scaled):            
            fig, ax = plt.subplots()
            ax.scatter(scaled[:, 0], scaled[:, 1], c='r')
            fig.tight_layout()
            return fig

        image_summary = draw_scatter(scaled)
        all_summaries = tf.summary.merge_all()

        writer = tf.summary.FileWriter('./log', sess.graph)

        summary = sess.run(all_summaries, feed_dict={scale: 2.})
        writer.add_summary(summary, global_step=0)
