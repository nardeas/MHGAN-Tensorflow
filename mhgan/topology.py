import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
import numpy as np

from mhgan.ops import gelu, gelu2

class Discriminator(object):
    '''
    Standard DCGAN discriminator with GELU activations.
    We use xavier initializer here with random normal as it's known to work best
    with sigmoidal activation functions (since we're using sigmoid based GELU
    approximation).
    '''
    def __init__(self, batch_norm=False, name='GAN/discriminator'):
        self.batch_norm = batch_norm
        self.name = name

    def __call__(self, x, reuse_vars=True):
        with tf.variable_scope('GAN/discriminator') as var_scope:
            reuse_vars and var_scope.reuse_variables()

            # Make sure X is N samples x Height x Width x Channels
            conv1 = tcl.conv2d(
                x, 64, [4, 4], [2, 2],
                weights_initializer=tcl.xavier_initializer(uniform=False),
                activation_fn=tf.identity
            )
            if self.batch_norm:
                conv1 = tcl.batch_norm(conv1)
            conv1 = gelu(conv1)

            conv2 = tcl.conv2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tcl.xavier_initializer(uniform=False),
                activation_fn=tf.identity
            )
            if self.batch_norm:
                conv2 = tcl.batch_norm(conv2)
            conv2 = gelu(conv2)
            conv2 = tcl.flatten(conv2)

            fc1 = tcl.fully_connected(
                conv2, 1024,
                weights_initializer=tcl.xavier_initializer(uniform=False),
                activation_fn=tf.identity
            )
            fc1 = gelu(fc1)
            return tcl.fully_connected(
                fc1,
                1,
                activation_fn=tf.identity,
                scope='output'
            )

    @property
    def vars(self):
        return list(filter(
            lambda var: self.name in var.name,
            tf.global_variables()
        ))

class Generator(object):
    '''
    Standard DCGAN generator with GELU activation. We use a tanh based GELU
    approximation which is more accurate but slightly slower.
    '''
    def __init__(self, input_shape, output_shape, batch_norm=False, name='GAN/generator'):
        self.input_shape = np.asarray(input_shape)
        self.output_shape = np.asarray(output_shape)
        self.batch_norm = batch_norm
        self.name = name

        assert len(self.output_shape) == 3, 'output shape must have 3 dimensions'
        assert np.prod(self.output_shape) % 4 == 0, 'output shape must be divisible by 4'

    def __call__(self, z):
        with tf.variable_scope('GAN/generator') as scope:
            batch_size = tf.shape(z)[0]

            fc1 = tcl.fully_connected(
                z,
                1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            if self.batch_norm:
                fc1 = tcl.batch_norm(fc1)
            fc1 = gelu2(fc1)

            fc2 = tcl.fully_connected(
                fc1,
                np.asscalar(128 * np.prod(self.output_shape[:2] // 4)),
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc2 = tf.reshape(fc2, tf.stack([
                batch_size,
                *(self.output_shape[:2] // 4),
                128
            ]))
            if self.batch_norm:
                fc2 = tcl.batch_norm(fc2)
            fc2 = gelu2(fc2)

            conv1 = tcl.conv2d_transpose(
                fc2, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            if self.batch_norm:
                conv1 = tcl.batch_norm(conv1)
            conv1 = gelu2(conv1)

            conv2 = tcl.conv2d_transpose(
                conv1, 1, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.nn.tanh
            )
            return tf.reshape(conv2, tf.stack([
                batch_size,
                *self.output_shape
            ]), name='output')

    @property
    def vars(self):
        return list(filter(
            lambda var: self.name in var.name,
            tf.global_variables()
        ))
