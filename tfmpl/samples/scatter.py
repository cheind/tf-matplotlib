# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

from datetime import datetime
import tensorflow as tf
import numpy as np
import os

import tfmpl

if __name__ == '__main__':

    with tf.Session(graph=tf.Graph()) as sess:
        
        @tfmpl.figure_tensor
        def draw_scatter(scaled, colors): 
            '''Draw scatter plots. One for each color.'''  
            figs = tfmpl.create_figures(len(colors), figsize=(4,4))
            for idx, f in enumerate(figs):
                ax = f.add_subplot(111)
                ax.axis('off')
                ax.scatter(scaled[:, 0], scaled[:, 1], c=colors[idx])
                f.tight_layout()

            return figs  

        points = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(100, 2)).astype(np.float32))
        scale = tf.placeholder(tf.float32)        
        scaled = points*scale
       
        image_tensor = draw_scatter(scaled, ['r', 'g'])
        image_summary = tf.summary.image('scatter', image_tensor)
        all_summaries = tf.summary.merge_all()

        os.makedirs('log', exist_ok=True)
        now = datetime.now()
        logdir = "log/" + now.strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, sess.graph)

        summary = sess.run(all_summaries, feed_dict={scale: 2.})
        writer.add_summary(summary, global_step=0)
        writer.flush()
