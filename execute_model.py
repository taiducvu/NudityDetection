'''
Created on Dec 12, 2016

@author: taivu
'''

import tensorflow as tf
import numpy as np
import Dataset.data as dt
import math
from Classifiers import vng_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_dir', '/home/taivu/workspace/NudityDetection/Dataset',
                       """The path of folder consisting the checkpoint file""")

tf.flags.DEFINE_integer('num_examples', 1000,
                        """The number of examples used to evaluate the model""")

tf.flags.DEFINE_integer('eval_batch_size', 10,
                        """The number of examples in each batch""")

tf.flags.DEFINE_string('data_dir', '/home/taivu/workspace/NudityDetection/Dataset/nudity_test_set.tfrecords',
                       """The path of folder consisting the test set""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

       
def evaluate():
    with tf.Graph().as_default() as g:
        eval_images, real_lb = dt.data_input(FLAGS.data_dir, FLAGS.eval_batch_size, False)

        logits = vng_model.inference(eval_images)

        predict_label = tf.arg_max(logits, 1)
        
        init = tf.global_variables_initializer()
        # Load trained weights
        saver = tf.train.Saver(tf.global_variables())

        coord = tf.train.Coordinator()
        
        with tf.Session(config=tf.ConfigProto(
            log_device_placement = FLAGS.log_device_placement)) as sess:
            sess.run(init)
            tf.train.start_queue_runners(sess, coord)
            
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            else:
                print('No checkpoint file found')
                
            pre_label = sess.run([predict_label])            
            
            eval_img = eval_images.eval()
            
            real_label = real_lb.eval()        
            
        coord.request_stop()
        
        return eval_img, pre_label[0], real_label
        
def main(argv=None):
    evaluate()
    
if __name__ == '__main__':
    tf.app.run()
        
        
        