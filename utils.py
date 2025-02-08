""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    sampler = lambda x: random.sample(x, min(len(x), nb_samples))  # Avoid error

    for path in paths:
        files = os.listdir(path)
        print(f"Checking folder: {path} - {len(files)} images available, {nb_samples} needed")

        if len(files) < nb_samples:
            print(f"⚠️ WARNING: Not enough images in {path}. Found {len(files)}, expected {nb_samples}")

    return [(os.path.join(path, image), i) 
            for i, path in zip(labels, paths) 
            for image in sampler(os.listdir(path))]

import tensorflow as tf

def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.leaky_relu, max_pool_pad='VALID', residual=False):
    """ Perform conv, batch norm, nonlinearity, dropout, and max pool """

    # Define stride settings
    stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]

    # Apply convolution
    conv_output = tf.nn.conv2d(inp, cweight, no_stride if FLAGS.max_pool else stride, 'SAME') + bweight

    # Normalize (Batch Norm or Layer Norm)
    normed = normalize(conv_output, activation, reuse, scope)

    # Apply dropout (only during training)
    if FLAGS.train:  
        normed = tf.nn.dropout(normed, rate=0.3)  # Drop 30% of neurons

    # Apply max pooling if enabled
    if FLAGS.max_pool:
        normed = tf.nn.max_pool(normed, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=max_pool_pad)

    return normed


def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size
