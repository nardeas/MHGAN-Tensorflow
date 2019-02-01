import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
import numpy as np

class WGAN:

    def __init__(self, generator, discriminator, train_helper=None, L=10.):
        self.generator = generator
        self.discriminator = discriminator
        self.train_helper = train_helper

        self.z = tf.placeholder(
            tf.float32,
            shape=[None, *self.generator.input_shape],
            name='input_noise'
        )
        self.x = tf.placeholder(
            tf.float32,
            shape=[None, *self.generator.output_shape],
            name='input_real'
        )

        self.G = self.generator(self.z)
        self.D = self.discriminator(self.x, reuse_vars=False)
        self.Dg = self.discriminator(self.G, reuse_vars=True)

        self.G_loss = tf.reduce_mean(-self.Dg)
        self.D_loss = tf.reduce_mean(self.Dg - self.D) + self.gradient_penalty(L=L)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            all_vars = tf.global_variables()

            # Generator training operation
            self.G_train_op = tf.train.AdamOptimizer(
                learning_rate=2e-4,
                beta1=0.5,
                beta2=0.9
            ).minimize(
                self.G_loss,
                var_list=self.generator.vars
            )

            # Discriminator training operation
            self.D_train_op = tf.train.AdamOptimizer(
                learning_rate=2e-4,
                beta1=0.5,
                beta2=0.9
            ).minimize(
                self.D_loss,
                var_list=self.discriminator.vars
            )

        self.session = None

    def gradient_penalty(self, L):
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.G
        d_hat = self.discriminator(x_hat, reuse_vars=True)
        return tf.reduce_mean(
            tf.square(
                tf.sqrt(
                    tf.reduce_sum(
                        tf.square(
                            tf.gradients(d_hat, x_hat)[0]
                        ),
                        axis=1
                    )
                ) - 1.0
            ) * L)

    def train(self,
        sess,
        data_sampler,
        noise_sampler,
        batch_size=64,
        n_epochs=10,
        n_critic=5,
        n_accumulate=1):

        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.losses = []
        for epoch in range(1, n_epochs+1):
            try:

                # Train the discriminator for N iterations
                for _ in range(n_critic):
                    real_samples = data_sampler(batch_size)
                    noise_samples = noise_sampler(shape=[
                        batch_size,
                        *self.generator.input_shape
                    ])
                    _, d_loss = self.sess.run([self.D_train_op, self.D_loss], feed_dict={
                        self.x: real_samples,
                        self.z: noise_samples,
                    })

                # Train the generator for 1 iteration
                _, g_loss = self.sess.run([self.G_train_op, self.G_loss], feed_dict={
                    self.x: real_samples,
                    self.z: noise_sampler(shape=[
                        batch_size,
                        *self.generator.input_shape
                    ])
                })

                self.losses.append([ g_loss, d_loss ])

                # Optional: accumulate state for plotting loss or analyzing samples from G
                if epoch == 1 or epoch % n_accumulate == 0:
                    try:
                        self.train_helper and self.train_helper(
                            epoch=epoch,
                            state=(self.sess, np.asarray(self.losses), [
                                self.z,
                                self.G,
                                # Return noise sampler helper for analysis
                                lambda size: noise_sampler(shape=[
                                    size,
                                    *self.generator.input_shape
                                ])
                            ])
                        )
                    except Exception as e:
                        print(e)
            except KeyboardInterrupt:
                break
        return self.sess
