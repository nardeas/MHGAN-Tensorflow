import tensorflow as tf
import numpy as np

class MHGAN:
    '''
    Wraps your trained WGAN with generator and discriminator for enhanced
    output sampling.
    '''
    def __init__(self, wgan):
        self.generator_input_shape = wgan.generator.input_shape
        self.generator_output_shape = wgan.generator.output_shape
        self.generator_input_tensor = wgan.z
        self.generator_output_tensor = wgan.G
        self.discriminator_input_tensor = wgan.x
        self.fake_output_tensor = wgan.Dg
        self.real_output_tensor = wgan.D
        with tf.name_scope('MH'):
            self.c = tf.placeholder(tf.int32, [], name='total_count')
            self.k = tf.placeholder(tf.int32, [], name='k_count')
            self.u = tf.random_uniform([ self.c, self.k ])

            # Scores for calibration + generated samples
            self.scores = tf.reshape(
                # Add calibration scores from real discriminator output
                tf.concat([
                    tf.sigmoid(self.real_output_tensor),
                    tf.sigmoid(self.fake_output_tensor)
                ], 0),
                (self.c, self.k + 1)
            )

    def generate(self, sess, noise_sampler, count=1, squeeze=True):
        '''
        Draws <count> number of samples from Generator
        '''
        samples = sess.run(self.generator_output_tensor, feed_dict={
            self.generator_input_tensor: noise_sampler((
                count,
                *self.generator_input_shape
            ))
        })
        if squeeze:
            return samples.squeeze(axis=3)
        return samples

    def generate_enhanced(self, sess, data_sampler, noise_sampler, count=1, k=100, squeeze=True):
        '''
        Draws <count> number of enhanced samples from Generator with
        Metropolis-Hastings algorithm.
        '''
        # Draw samples and epsilon values, compute scores
        scores, epsilon, samples = sess.run([
                self.scores,
                self.u,
                self.generator_output_tensor
            ], feed_dict={
                self.generator_input_tensor: noise_sampler(shape=[
                    count * k,
                    *self.generator_input_shape
                ]),
                # Calibration scores from real data
                self.discriminator_input_tensor: data_sampler(
                    count
                ),
                self.c: count,
                self.k: k
            }
        )
        # Metropolis-Hastings GAN algorithm
        selected = []
        for i in range(count):
            x = 0
            for x_next in range(k):
                alpha = np.fmin(1., (1./scores[i][x] - 1.) / (1./scores[i][x_next] - 1.))
                if epsilon[i][x_next] <= alpha:
                    x = x_next
                # Avoid samples from calibration distribution
                x += int(x == 0)
            selected.append(samples[x])
        selected = np.asarray(selected)
        if squeeze and selected.ndim > 3:
            return selected.squeeze(axis=3)
        return selected
