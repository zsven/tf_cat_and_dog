import tensorflow as tf
from config import *

trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
constant = lambda value: tf.constant_initializer(value)


def _variable_normal(name, shape, initializer, dtype=tf.float32):
    var = tf.get_variable(name=name, shape=shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_decay(name, shape, initializer, dtype=tf.float32, decay=DECAY):
    var = _variable_normal(name, shape, initializer, dtype)
    if decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), decay, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(images, keep_prob, batch_size=BATCH_SIZE, n_class=N_CLASSES):
    """
    :param images: with shape=[batch_size, 208, 208, 3]
    :param batch_size: batch_size
    :param n_class: 2
    :return: out
    """
    # a simple VGG-16
    with tf.variable_scope('conv1') as scope:
        w1 = _variable_with_decay('weight', shape=[3, 3, 3, 64], initializer=trunc_normal(0.1), decay=None)
        b1 = _variable_normal('biases', shape=[64], initializer=constant(0.0))
        conv1 = tf.nn.conv2d(images, w1, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.bias_add(conv1, bias=b1)
        conv1 = tf.nn.relu(conv1, name=scope.name) # 208x208x64

    with tf.variable_scope('pool1') as scope:
        # 104x104x64
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2') as scope:
        w2 = _variable_with_decay('weight', shape=[3, 3, 64, 128], initializer=trunc_normal(0.01))
        b2 = _variable_normal('biases', shape=[128], initializer=constant(0.0))
        conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.bias_add(conv2, bias=b2)
        conv2 = tf.nn.relu(conv2, name=scope.name) # 104x104x128

    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2') # 52x52x128

    with tf.variable_scope('conv3'):
        w3 = _variable_with_decay('weight', shape=[3, 3, 128, 256], initializer=trunc_normal(0.01))
        b3 = _variable_normal('biases', shape=[256], initializer=constant(0.0))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool2, w3, strides=[1, 1, 1, 1], padding='SAME'), b3), name=scope.name)
        w3 = _variable_with_decay('weight:1', shape=[3, 3, 256, 256], initializer=trunc_normal(0.01))
        b3 = _variable_normal('biases:1', shape=[256], initializer=constant(0.0))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, w3, strides=[1, 1, 1, 1], padding='SAME'), b3), name=scope.name)

    with tf.variable_scope('pool4'):
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3') # 26x26x256

    with tf.variable_scope('conv4') as scope:
        w4 = _variable_with_decay('weight', shape=[3, 3, 256, 512], initializer=trunc_normal(0.01))
        b4 = _variable_normal('biases', shape=[512], initializer=constant(0.0))
        conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool3, w4, strides=[1, 1, 1, 1], padding='SAME'), b4), name=scope.name)
        w4 = _variable_with_decay('weight:1', shape=[3, 3, 512, 512], initializer=trunc_normal(0.01))
        b4 = _variable_normal('biases', shape=[512], initializer=constant(0.0))
        conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4, w4, strides=[1, 1, 1, 1], padding='SAME'), b4), name=scope.name) # 26 x 26 x512

    with tf.variable_scope('pool4') as scope:
        pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4') # 13 * 13 * 512

    with tf.variable_scope('conv5') as scope:
        w5 = _variable_with_decay('weight', shape=[3, 3, 512, 512], initializer=trunc_normal(0.01))
        b5 = _variable_normal('biases', shape=[512], initializer=constant(0.0))
        conv5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool4, w5, strides=[1, 1, 1, 1], padding='SAME'), bias=b5), name=scope.name)
        w5 = _variable_with_decay('weigh:1', shape=[3, 3, 512, 512], initializer=trunc_normal(0.01))
        b5 = _variable_normal('biases', shape=[512], initializer=constant(0.0))
        conv5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv5, w5, strides=[1, 1, 1, 1], padding='SAME'), b5), name=scope.name)

    with tf.variable_scope('pool5') as scope:
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5') #7 * 7 * 512

    with tf.variable_scope('fc6') as scope:
        w6 = _variable_with_decay('fc', shape=[7, 7, 512, 4096], initializer=trunc_normal(0.01), decay=None)
        b6 = _variable_normal('biases', shape=[4096], initializer=constant(0.0))
        fc6 = tf.nn.bias_add(tf.nn.conv2d(pool5, w6, strides=[1, 1, 1, 1], padding='VALID'), bias=b6)
        fc6 = tf.nn.dropout(fc6, keep_prob=keep_prob)

    with tf.variable_scope('fc7') as scope:
        w7 = _variable_with_decay('fc', shape=[1, 1, 4096, 4096], initializer=trunc_normal(0.01), decay=None)
        b7 = _variable_normal('biases', shape=[4096], initializer=constant(0.0))
        fc7 = tf.nn.bias_add(tf.nn.conv2d(fc6, w7, strides=[1, 1, 1, 1], padding='VALID'), b7)
        fc7 = tf.nn.dropout(fc7, keep_prob=keep_prob)

    with tf.variable_scope('fc8') as scope:
        w8 = _variable_with_decay('fc', shape=[1, 1, 4096, n_class], initializer=trunc_normal(0.01), decay=None)
        b8 = _variable_normal('biases', shape=[n_class], initializer=constant(0.0))
        fc8 = tf.nn.bias_add(tf.nn.conv2d(fc7, w8, strides=[1, 1, 1, 1], padding='VALID'), b8)
    net = tf.squeeze(fc8, [1, 2])
    return net


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection('losses'))
    return loss


def training(loss, learning_rate=LEARNIMG_RATE):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step')
        train_op = optimizer.minimize(loss=loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    correct = tf.cast(correct, tf.float32)
    accuracy = tf.reduce_mean(correct)
    return accuracy