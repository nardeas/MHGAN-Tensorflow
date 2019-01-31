import tensorflow as tf
import math

def gelu(x):
    with tf.name_scope('activation/gelu') as scope:
        c_t = tf.constant(1.702, name='sigmoid_approx_const')
        return tf.multiply(
            x,
            tf.sigmoid(tf.multiply(c_t, x))
        )

def gelu2(x):
    pi_2_sq = math.sqrt(2. / math.pi)
    with tf.name_scope('activation/gelu_2') as scope:
        c_t = tf.constant(.044715, name='tanh_approx_const')
        return tf.multiply(
            tf.multiply(.5, x),
            tf.add(1., tf.tanh(
                tf.multiply(
                    pi_2_sq,
                    tf.add(x, tf.multiply(c_t, tf.pow(x, 3)))
                )
            ))
        )
