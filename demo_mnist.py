import matplotlib.pyplot as plt
from keras.datasets import mnist
from helpers import *
from mhgan import *

np.random.seed(1)

def mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 1: Normalize our inputs to be in the range[-1, 1]
    # 2: Expand dimensions so we have 1 channel
    x_train = np.expand_dims((x_train.astype(np.float32) - 127.5)/127.5, 3)
    # Return real samples
    return x_train

def create_images_from_samples(samples, dimensions=(5, 5), epoch='latest'):
    fig = plt.figure(figsize=dimensions)
    for i in range(samples.shape[0]):
        plt.subplot(dimensions[0], dimensions[1], i+1)
        plt.imshow(samples[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    fig.text(0.42, 0.93, 'Epoch {}'.format(epoch), fontsize=12)
    plt.savefig('output/images/generated-{}.png'.format(epoch))
    plt.savefig('output/images/generated-latest.png')
    plt.show()

def create_accumulator(sample_nth=10, **summary_args):
    
    # Predefined noise vector
    noise = create_noise_sampler()([25, 100])
    
    add_summary = create_summary_helper(**summary_args)
    
    def train_accumulator(epoch, state):
        sess, losses, [z, g, z_sampler] = state

        if epoch == 1 or epoch % sample_nth == 0:
            print('Generating sample images...')

            # Generate some samples
            samples = sess.run(g, feed_dict={z: noise})

            # Remove channel dimension
            samples = samples.squeeze(axis=3)

            # Save samples as images
            create_images_from_samples(samples, epoch=epoch)
            
        add_summary(epoch, losses)
        print('Epoch {} losses G={:.6f} D={:.6f}'.format(epoch, losses.T[0][-1], losses.T[1][-1]))
    return train_accumulator

def create_data_sampler(data, subset_size=None):
    subset_size = subset_size or len(data)
    def sampler(batch_size):
        return data[:subset_size][np.random.permutation(subset_size)][:batch_size]
    return sampler

def create_noise_sampler():
    def sampler(shape):
        return np.random.normal(0, 1, shape)
    return sampler

if __name__ == '__main__':
    real_data = mnist_data()

    sess = tf.Session()

    noise_dimensions = [ 100 ]
    real_dimensions = real_data.shape[1:]

    noise_sampler = create_noise_sampler()
    data_sampler = create_data_sampler(real_data, subset_size=2000)

    gan = WGAN(
        Generator(input_shape=noise_dimensions, output_shape=real_dimensions),
        Discriminator()
    )
    gan.test_accumulate = create_accumulator(
        sess=sess,
        output_path='output/summaries/wgan'
    )
    gan.train(
        sess,
        data_sampler,
        noise_sampler,
        batch_size=32,
        n_epochs=100,
        n_accumulate=1
    )
