'''
Created on Dec 19, 2016

@author: taivu
'''
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.inception_v2 import inception_v2
from tensorflow.contrib.slim.python.slim.nets.inception_v2 import inception_v2_arg_scope
from PIL import Image
import numpy as np


def test_program():
    with tf.Graph().as_default():
        slim = tf.contrib.slim
        
        checkpoint_dir = '/home/taivu/workspace/NudityDetection/Trained_weight'
        
        
        input_tensor = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input_image')
        scaled_input_tensor = tf.scalar_mul((1.0/255), input_tensor)
        scaled_input_tensor = tf.sub(scaled_input_tensor, 0.5)
        scaled_input_tensor = tf.mul(scaled_input_tensor, 2.0)
         
        arg_scope = inception_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = inception_v2(scaled_input_tensor, is_training = False)
                
        init = tf.global_variables_initializer()
        
        saver = tf.train.Saver(tf.global_variables())
        
        coord = tf.train.Coordinator()
        
        with tf.Session() as sess:            
                
            sess.run(init)
            
            threads = tf.train.start_queue_runners(sess, coord)
            
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            im = Image.open('/home/taivu/workspace/NudityDetection/dog.jpeg').resize((224, 224))
            im = np.array(im)
            im = im.reshape(1, 224, 224, 3)
            
            predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={
                input_tensor: im})
               
            print(np.max(predict_values), np.max(logit_values))
            
        coord.request_stop()