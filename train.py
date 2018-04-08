from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from config import *
import model
import tensorflow as tf
import input_data as id
import os


def train():
    train_batch, label_batch = id.read_and_save()
    train_batch = tf.cast(train_batch, dtype=tf.float32)
    label_batch = tf.cast(label_batch, dtype=tf.int64)

    logits = model.inference(train_batch, label_batch)
    loss = model.losses(logits, label_batch)
    op = model.training(loss=loss)
    accuracy = model.evaluation(logits, label_batch)
    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(LOG_DIR, graph=sess.graph)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runner(sess=sess, coord=coord)
        try:
            for step in range(MAX_STEP):
                _, train_loss, train_acc = sess.run([op, loss, accuracy])
                if step % 50 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f' % (step, train_loss, train_acc))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)
                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(LOG_DIR, "model.ckpt")
                    saver.save(sess, checkpoint_path)
        except tf.errors.OutOfRangeError:
            print('An error occur')
        finally:
            coord.request_stop()
        coord.join(threads=threads)

if __name__ == '__main__':
    train()