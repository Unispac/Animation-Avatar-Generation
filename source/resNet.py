import tensorflow as tf
import numpy as np
from tools import *

def conv2(batch_input, kernel=3, output_channel=64, stride=1,scope='conv'):
    with tf.variable_scope(scope):
        return  tf.contrib.layers.convolution2d(
            batch_input, output_channel, [kernel, kernel], [stride, stride],
            padding='same',
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            activation_fn=None
            )
        

def conv2d_sn(ibatch_input, kernel=3, output_channel=64, stride=1, scope='conv'):
  with tf.variable_scope(scope):
        w = tf.get_variable(scope+"kernel",shape=[kernel, kernel, ibatch_input.get_shape()[-1], output_channel])
        net = lrelu(tf.nn.conv2d(input=ibatch_input, filter=spectral_norm(w,"sn3"), strides=[1,stride,stride,1],padding="SAME"))
        return net

def residual_block(inputs, output_channel, stride, scope, train = True):
    with tf.variable_scope(scope):
        net = conv2(inputs, 3, output_channel, stride, scope='conv_1')
        net = tf.layers.batch_normalization(net, training=train)
        net = tf.nn.relu(net)
        net = conv2(net, 3, output_channel, stride, scope='conv_2')
        net = tf.layers.batch_normalization(net, training=train)
        net = net + inputs

    return net

def discriminator_block(inputs, output_channel, kernelSize, stride, scope):
        res = inputs

        with tf.variable_scope(scope):
            #net = conv2(inputs, 3, output_channel, stride, use_bias=False, scope='conv_1')
            net = conv2d_sn(inputs, kernelSize, output_channel, stride, scope='conv_1')
            #net = conv2(inputs, kernel_size, output_channel, stride, use_bias=False, scope='conv1')
            net = lrelu(net, 0.2)

            net = conv2d_sn(inputs, kernelSize, output_channel, stride, scope='conv_2')
            #net = conv2(net, kernel_size, output_channel, stride, use_bias=False, scope='conv2')
            net = net + res
            net = lrelu(net, 0.2)

        return net

def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output

def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)
