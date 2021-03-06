'''
Created on Dec 12, 2016

@author: taivu
'''

import tensorflow as tf
import Dataset.data as dt
import csv
import math
import numpy as np
import os
from Classifiers import vng_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_dir', '/home/taivu/workspace/NudityDetection/Dataset',
                       """The path of folder consisting the checkpoint file""")

tf.flags.DEFINE_string('output_dir', '/home/taivu/workspace/NudityDetection/Output',
                       """The path of folder consisting the a csv output-file""")

tf.flags.DEFINE_integer('num_examples', 80,
                        """The number of examples used to evaluate the model""")

tf.flags.DEFINE_integer('eval_batch_size', 180,
                        """The number of examples in each batch""")

tf.flags.DEFINE_string('data_dir', '/home/taivu/workspace/AddPic',
                       """The path of folder consisting the test set""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

       
def evaluate(data_dir=None, real_time = True):
    with tf.Graph().as_default():
        
        if real_time:
            if data_dir is not None:
                eval_images, ls_filename = dt.data_input(data_dir, FLAGS.eval_batch_size, False, False)
            
            else:
                eval_images, ls_filename = dt.data_input(FLAGS.data_dir, FLAGS.eval_batch_size, False, False)
            
        else:   
            if data_dir is not None:
                eval_images, real_lb = dt.data_input(data_dir, FLAGS.eval_batch_size, False)
                print(eval_images.get_shape())
            else:
                eval_images, real_lb = dt.data_input(FLAGS.data_dir, FLAGS.eval_batch_size, False)
                print(eval_images.get_shape())
                
        logits = vng_model.inference(eval_images)

        predict_label = tf.arg_max(logits, 1)
        
        init = tf.global_variables_initializer()
        # Load trained weights
        saver = tf.train.Saver(tf.global_variables())

        coord = tf.train.Coordinator()
        
        with tf.Session(config=tf.ConfigProto(
            log_device_placement = FLAGS.log_device_placement)) as sess:
            sess.run(init)
            
            threads = tf.train.start_queue_runners(sess, coord)
            
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            else:
                print('No checkpoint file found')
            
            if real_time:
                num_examples = len([name_file for name_file in os.listdir(FLAGS.data_dir)])
                num_iter =  int(math.ceil(float(num_examples) / FLAGS.eval_batch_size))
                
                path_output = os.path.join(FLAGS.output_dir,'output.csv')
                with open(path_output, "wb") as f:
                    writer = csv.writer(f)
                    for idx in range(num_iter):
                        eval_img, pre_label, ls_name = sess.run([eval_images, predict_label, ls_filename])
                        
                        if (idx + 1) * FLAGS.eval_batch_size <= num_examples:
                            ls_name = [name_file.split('/')[-1] for name_file in ls_name]
                            result_model = np.column_stack((np.array(ls_name), np.array(pre_label)))
                        
                        else:
                            if num_examples - idx * FLAGS.eval_batch_size > 0:
                                last_element = num_examples - idx * FLAGS.eval_batch_size
                            else:
                                last_element = num_examples
                            
                            ls_name = [name_file.split('/')[-1] for name_file in ls_name]
                            result_model = np.column_stack((np.array(ls_name)[0 : last_element], 
                                                            np.array(pre_label)[0 : last_element]))
                            
                        writer.writerows(result_model)               

                real_label = None
                
            else:
                eval_img, pre_label, real_label = sess.run([eval_images, predict_label, real_lb])      
                
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=5)
            sess.close()

        return eval_img, pre_label, real_label
        
def main(argv=None):
    evaluate()
    
if __name__ == '__main__':
    tf.app.run()
        
        