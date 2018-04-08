from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os
from PIL import Image
from config import *


def read_data(data_dir=TRAIN_DIR):
    cats = []
    dogs = []
    label_cats = []
    label_dogs = []
    for file in os.listdir(data_dir):
        name = file.split('.')
        if name[0] == 'cat':
            cats.append(data_dir + file)
            label_cats.append(0)
        else:
            dogs.append(data_dir + file)
            label_dogs.append(1)
    images = np.hstack((cats, dogs))
    labels = np.hstack((label_cats, label_dogs))

    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)

    images = list(temp[:, 0])
    labels = list(temp[:, 1])
    labels = [i for i in labels]
    return images, labels


def check_value(value):
    if not isinstance(value, list):
        value = [value]
    return value


def _to_bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=check_value(value)))


def _to_int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=check_value(value)))


def save_tfrecords(images, labels, save_dir=RECORD_DIR):
    if np.shape(images)[0] != len(labels):
        raise ValueError('Images size %d not equal to labels size %d' % (np.shape(images)[0], len(labels)))
    writer = tf.python_io.TFRecordWriter(save_dir)
    try:
        for i in range(len(labels)):
            image = Image.open(images[i])
            image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            image = np.array(image)
            image_raw = image.tostring()
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _to_bytes_feature(image_raw),
                'label': _to_int_feature(label)
            }))
            writer.write(example.SerializeToString())
    except IOError:
        print('Could not read')
    finally:
        writer.close()


def read_and_save(batch_size=BATCH_SIZE, capacity=CAPACITY, num_threads=NUM_THREADS,
                  data_dir=TRAIN_DIR, save_dir=RECORD_DIR):
    """
    read dogs_vs_cats dataset and save to a .tfrecords file 
    """
    if not os.path.exists(save_dir):
        images, labels = read_data(data_dir)
        save_tfrecords(images, labels)
    filename_queue = tf.train.string_input_producer([save_dir])
    reader = tf.TFRecordReader()
    _, examples = reader.read(filename_queue)
    features = tf.parse_single_example(examples, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string)
    })
    image = tf.decode_raw(features['image'], tf.uint8)
    label = tf.cast(features['label'], tf.float32)
    image = tf.reshape(image, [IMAGE_WIDTH, IMAGE_HEIGHT, 3])
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size,
                                              num_threads=num_threads, capacity=capacity)
    return image_batch, tf.reshape(label_batch, [batch_size])

