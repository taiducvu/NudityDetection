'''
Created on Nov 24, 2016

@author: cpu11757
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import model.classifiers.vng_model as vng_md
from model.datasets.data import data_input

import time
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/cpu11757/workspace/Nudity_Detection/src/model/datasets',
                           """Direction where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('batch_size', 128, """Number of examples each a batch""")

def set_train_flow(flag, value):
    flag.assign(value)   

def train():
    with tf.Graph().as_default():
        flag_is_training = True
        global_step = tf.Variable(0, trainable=False)
        flag_training = tf.Variable(True, trainable=False)
        
        tr_images, tr_labels = data_input('/home/cpu11757/workspace/Nudity_Detection/src/model/datasets/vng_dataset.tfrecords', 128)
        
        val_images, val_labels = data_input('/home/cpu11757/workspace/Nudity_Detection/src/model/datasets/vng_dataset_test.tfrecords', 1000, False)
        
        #with tf.variable_scope("image_filters") as scope:
        
        #images = tf.placeholder(tf.float32, shape=[None, 34, 34, 3], name='input_image')
        #labels = tf.placeholder(tf.float32, shape=[None,], name='label')
        validation_flow = flag_training.assign(False)
        training_flow = flag_training.assign(True)
        
        
        
        images, labels = tf.cond(flag_training, 
                           lambda: (tr_images, tr_labels),
                           lambda: (val_images,val_labels))
        
        #logits = tf.cond(flag_is_training, vng_md.inference(tr_images), vng_md.inference(val_images))
        logits = vng_md.inference(images)

        #val_logits = vng_md.inference(val_images)
        
        #loss = vng_md.loss(logits, tr_labels)
        loss = vng_md.loss(logits, labels)
        
        train_op = vng_md.train(loss, global_step)
        
        saver = tf.train.Saver(tf.all_variables())        
        
        summary_op = tf.merge_all_summaries()
        
        correct_predict = tf.equal(tf.cast(tf.arg_max(logits, 1),tf.int32), labels)
        
        val_accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        
        init = tf.initialize_all_variables()
        
        
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement = FLAGS.log_device_placement))
        
        sess.run(init)
        
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess, coord)
        
        
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
        
        
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            ##
            #tr_img, tr_lb = sess.run([tr_images, tr_labels])
            #feed_dict = {images:tr_img, labels:tr_lb}
            ##

            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
            
            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                
                format_str = ('%s: step :%d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))
                
                #sess.run([validation_flow])
                sess.run(validation_flow)
                va_acc = sess.run([val_accuracy])
                #print ('Validation accuracy: %.2f'%va_acc)
                print(va_acc)
                sess.run(training_flow)
                

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                #val_acc = sess.run(val_accuracy)
                #print('Validation accuracy: %.2f'%val_acc)
            
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step = step)

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()