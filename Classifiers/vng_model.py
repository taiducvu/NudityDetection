'''
Created on Nov 23, 2016

@author: taivu
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Dataset.data as data

import re
import tensorflow  as tf

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch""")
tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")
#tf.app.flags.DEFINE_string('data_dir', '/tmp/data', """Path to the nudity dataset""")

TOWER_NAME = 'tower'

# GLOBAL CONSTANT
NUM_CLASSES = data.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = data.NUM_EXAMPLES_PER_EPOCH_TRAIN

MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

def _activation_summary(x):
    """
    """
    
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    
    
def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory
    
    Args:
    name: name of the variable
    shape:
    initializer: initializer for variable
    
    Returns:
    Variable tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, dtype, initializer)
    return var
    
def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay:
    
    Note that the Variable is initialized with a truncated normal distribution
    A weight decay is added only if one is specified.
    
    Args:
    name: name of the variable
    shape: the shape of weights
    stddev: standard deviation of a truncated Gaussian
    wd: add L2loss weight decay multiplied by this float.
    
    Returns:
    
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(name,
                           shape,
                           tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    
    return var

def inputs(eval_data):
    # Following cifar10_train
    pass

# This function preference
def inference(images):
    """Build the VNG model
    
    Args:
    """
    num_example = images.get_shape()[0].value
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)
    
    # pool1    
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)
        
    # norm 2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    
    # conv3: Experimental Layers
    #with tf.variable_scope('conv3') as scope:
    #    kernel = _variable_with_weight_decay('weights',
    #                                         shape = [3, 3, 64, 64], 
    #                                         stddev = 5e-2,
    #                                         wd = 0.0)
        
    #    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    #    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    #    bias = tf.nn.bias_add(conv, biases)
    #    conv3 = tf.nn.relu(bias, name = scope.name)
    #    _activation_summary(conv3)
    
    #norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75, name='norm3')
    
    # local3
    with tf.variable_scope('local3') as scope:
        # reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        #reshape = tf.reshape(pool2, [-1, 16 * 16 * 64]) # Input's shape: 64x64x3
        reshape = tf.reshape(pool2, [-1, 9 * 9 * 64])
        
        # dim = reshape.get_shape()[1].value
        # Neuron cu: 384
        #weights = _variable_with_weight_decay('weights', shape=[16 * 16 * 64, 384], stddev=0.04, wd=0.004)
        weights = _variable_with_weight_decay('weights', shape=[9 * 9 * 64, 384], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)
        
    # local4
    with tf.variable_scope('local4') as scope:
        #Neuron cu 192
        weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)
    
    # Tang moi them vao    
    #with tf.variable_scope('local5') as scope:
    #    weights = _variable_with_weight_decay('weights', shape=[200, 100], stddev = 0.04, wd=0.004)
    #    biases = _variable_on_cpu('biases', [100], tf.constant_initializer(0.1))
    #    local5 = tf.nn.relu(tf.matmul(local4, weights) + biases, name = scope.name)
    #    _activation_summary(local5)
        
    # Softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', shape=[192, NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)
    
    return softmax_linear

def loss(logits, labels):
    """
    """
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
    """
    """
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    
    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))
        
    return loss_averages_op

def train(total_loss, global_step):
    
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    
    tf.scalar_summary('learning_rate', lr)
    
    loss_averages_op = _add_loss_summaries(total_loss)
    
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
        
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
        
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)
        
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')
    
    return train_op
    
