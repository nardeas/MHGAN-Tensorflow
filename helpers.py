import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def images_from_samples(samples, dimensions=(5, 5), epoch=None, save=True):
    # Remove channel dimension if present
    if samples.ndim > 3 and samples.shape[-1] == 1:
        samples = samples.squeeze(axis=3)

    fig = plt.figure(figsize=dimensions)
    for i in range(samples.shape[0]):
        plt.subplot(dimensions[0], dimensions[1], i+1)
        plt.imshow(samples[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')

    # Approximate top-center for title text
    epoch and fig.text(0.42, 0.93, 'Epoch {}'.format(epoch), fontsize=12)

    if save:
        plt.savefig('output/images/generated-{}.png'.format(epoch))
        plt.savefig('output/images/generated-latest.png')
    plt.show()

def create_summary_helper(sess, output_path):

    with tf.name_scope('generator'):
        generator_loss_history = tf.placeholder(
            tf.float32,
            [ None ],
            name='loss_history_placeholder'
        )
        generator_mean_loss = tf.reduce_mean(
            generator_loss_history,
            name='mean_loss_placeholder'
        )
        generator_summary = tf.summary.merge([
            tf.summary.scalar('loss', generator_loss_history[-1]),
            tf.summary.scalar('mean_loss', generator_mean_loss),
            tf.summary.histogram('loss_history', generator_loss_history)
        ])

    with tf.name_scope('discriminator'):
        discriminator_loss_history = tf.placeholder(
            tf.float32,
            [ None ],
            name='loss_history_placeholder'
        )
        discriminator_mean_loss = tf.reduce_mean(
            discriminator_loss_history,
            name='mean_loss_placeholder'
        )
        discriminator_summary = tf.summary.merge([
            tf.summary.scalar('loss', discriminator_loss_history[-1]),
            tf.summary.scalar('mean_loss', discriminator_mean_loss),
            tf.summary.histogram('loss_history', discriminator_loss_history)
        ])

    g_writer = tf.summary.FileWriter(
        output_path + '/generator',
        sess.graph
    )
    d_writer = tf.summary.FileWriter(
        output_path + '/discriminator',
        #sess.graph
    )

    def add_summaries(epoch, accumulate_losses):
        g_writer.add_summary(sess.run(
            generator_summary,
            feed_dict={
                generator_loss_history: accumulate_losses.T[0]
            }),
            epoch
        )
        d_writer.add_summary(sess.run(
            discriminator_summary,
            feed_dict={
                discriminator_loss_history: accumulate_losses.T[1]
            }),
            epoch
        )

    return add_summaries

def create_train_helper(
    sample_count=25,
    sample_nth=10,
    sample_save=True,
    summaries=True,
    **summary_args):

    # Summary helper for Tensorboard
    add_summary = lambda *a: None
    if summaries:
        add_summary = create_summary_helper(**summary_args)

    def train_helper(epoch, state):
        sess, losses, (generator_input, generator_output, noise_sampler) = state

        # NOTE: Feel free to plot losses, or use Tensorboard with summaries
        # losses

        # Predefined noise vector for comparison
        if train_helper.noise is None:
            train_helper.noise = noise_sampler(sample_count)

        # Generate some samples and save as images
        if epoch == 1 or epoch % sample_nth == 0:
            print('Info: Generating sample images...')
            grid_size = int(np.sqrt(sample_count))
            images_from_samples(
                epoch=epoch,
                save=sample_save,
                dimensions=(grid_size, grid_size),
                samples=sess.run(generator_output, feed_dict={
                    generator_input: train_helper.noise
                })
            )
        add_summary(epoch, losses)
        print('Training: epoch {} losses => generator={:.6f}, discriminator={:.6f}'.format(
            epoch,
            losses.T[0][-1],
            losses.T[1][-1]
        ))
    train_helper.noise = None
    return train_helper
