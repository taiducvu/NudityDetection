'''
Created on Nov 22, 2016

@author: taivu
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from _collections import defaultdict

CENTRAL_FRACTION = 0.875

import glob
import tensorflow as tf
import os
from scipy.misc import imread
import numpy as np
import csv
from shutil import copyfile

NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_TRAIN = 5000

def read_data(filename_queue):
    pass

def normalize_name_file(dir_path, start_num, pattern):
    idx = start_num
    for pathAndFileName in glob.iglob(os.path.join(dir_path, '*.jpg')):
        _, ext = os.path.splitext(os.path.basename(pathAndFileName))
        os.rename(pathAndFileName, os.path.join(dir_path, 'd_%d' % idx + ext))
        idx += 1
    
    
def preprocess_image(image, height, width, scope=None):
    """Pre-process one image for training or evaluation
    Arg:
    image: 3-D Tensor [height, width, channels] with the image,
    height: integer, image expected height.
    width: integer, image expected width.
    
    Returns:
    3-D float Tensor containing an appropriately scaled image
    """
    
    with tf.name_scope(scope, 'preprocess_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        
        if CENTRAL_FRACTION:
            image = tf.image.central_crop(image, CENTRAL_FRACTION)
            
        if height and width:
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
            image = tf.squeeze(image, [0])

        return image

def process_raw_dataset(path_csv, processed_data):
    """
     In this function, we read a csv file to separate the data-set into two parts called "nudity"
    and "normal". We, then, copy the nude images into the "nudity" folder and the other into the
    "normal" folder.
    
    Note: This function is only executed one time
    
    Arg:
    path_csv: the path of csv file
    processed_data: the path of the folder consists of 
    Return:
    
    """
    
    labels = defaultdict(list)
    with open(path_csv, 'rb') as csvfile:
        stream_data = csv.DictReader(csvfile, delimiter=',')
        for row in stream_data:
            for (k, v) in row.items():
                labels[k].append(v)
                
    
    labels['ID'] = np.array(map(int, labels['ID']))
    labels['Nude'] = np.array(map(int, labels['Nude']))
    
    mark_nude = labels['Nude'] == 1
    mark_normal = labels['Nude'] == 0
    
    normal_id = labels['ID'][mark_normal]
    nude_id = labels['ID'][mark_nude]
    
    path_dataset = os.path.join(processed_data, 'nudity_dataset')
    for count, id in zip(range(len(nude_id)), nude_id):
        if count < 1000:      
            copyfile(os.path.join(path_dataset, '%d.jpg' % id), os.path.join(processed_data, 'train/nude/%d.jpg' % id))
        
        else:
            copyfile(os.path.join(path_dataset, '%d.jpg' % id), os.path.join(processed_data, 'test/nude/%d.jpg' % id))
    
    for count, id in zip(range(len(normal_id)), normal_id):
        if count < 1000:
            copyfile(os.path.join(path_dataset, '%d.jpg' % id), os.path.join(processed_data, 'train/normal/%d.jpg' % id))
        
        else:
            copyfile(os.path.join(path_dataset, '%d.jpg' % id), os.path.join(processed_data, 'test/normal/%d.jpg' % id))
    
    return labels
    

def normalize_dataset_type(dir_path, num_samples, is_train=None):
    """Build a standard structure of the data-set
    Args:
    
    Return:
    """
        
    if is_train is not None:
        pass
    
    dataset = dict.fromkeys([''])
    for file_id in range(num_samples):
        image = imread(os.path.join(dir_path, '%s.jpg' % str(file_id)))
        dataset.append(image)

    dataset = np.array(dataset)
    return dataset

def generate_standard_dataset(dir_path):
    """ This function normalize raw data-set into a standard data-set
    with respect to our model.
    """
    filenames = []
    
    # Normal Training data
    for pathAndFileName in glob.iglob(os.path.join(dir_path, '*.jpg')):
        filenames.append(pathAndFileName)
    
    filename_queue = tf.train.string_input_producer(filenames, shuffle=None)
    
    reader = tf.WholeFileReader()
    
    _, value = reader.read(filename_queue)
    
    image = tf.image.decode_jpeg(value, 3)
    
    image = preprocess_image(image, height=34, width=34)
    
    return image, filenames

def _int64_feature(value):
    """
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(data_dir, dataset, labels, name):
    """Converts a dataset to tfrecords."""
    images = dataset
    labels = labels
    num_examples = dataset.shape[0]
    
    rows, cols, depth = dataset[0].shape
    
    filename = os.path.join(data_dir, name + '.tfrecords')
    
    writer = tf.python_io.TFRecordWriter(filename)
    
    for idx in range(num_examples):
        image_raw = images[idx].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(rows),
                    'width': _int64_feature(cols),
                    'depth': _int64_feature(depth),
                    'label': _int64_feature(int(labels[idx])),
                    'image_raw': _bytes_feature(image_raw)
                }))
        writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(filename_queue):
    """
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
    serialized_example,
    features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64)
        })
    
    image = tf.decode_raw(features['image_raw'], tf.float32)
    image = tf.reshape(image, [34, 34, 3])
    label = tf.cast(features['label'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)
    return image, label, height, width, depth


def data_input(data_dir, batch_size, is_training=True, normalized_data = True):
    """
    """
    
    if normalized_data:
        filename_queue = tf.train.string_input_producer([data_dir], num_epochs=None)
        image, label, _, _, _ = read_and_decode(filename_queue)
        if is_training:
            images_batch, labels_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                capacity=4000,
                min_after_dequeue=80,
                name='input_train_data'
            )
        else:
            images_batch, labels_batch = tf.train.batch([image, label],
                                                        batch_size=batch_size,
                                                        capacity=2000,
                                                        name='input_data_validation')
        
        return images_batch, labels_batch
    
    else:
        #path_new = os.path.join(data_dir, '*.jpg')
        filenames = []    
    
        for pathAndFileName in glob.iglob(os.path.join(data_dir, '*.jpg')):
            filenames.append(pathAndFileName)
        
        filename_queue = tf.train.string_input_producer(filenames,  shuffle=None)
        
        filename = filename_queue.dequeue()

        #img_reader = tf.WholeFileReader()

        #_, img_file = img_reader.read(filename_queue)
        
        img_file = tf.read_file(filename) 
        
        image = tf.image.decode_jpeg(img_file, 3)

        image = preprocess_image(image, height = 34, width = 34)
        
        images, ls_name = tf.train.batch([image, filename],
                                batch_size = batch_size,
                                capacity = 80,
                                allow_smaller_final_batch=True,
                                name='input_stream_data')    
        return images, ls_name