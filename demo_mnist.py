import matplotlib.pyplot as plt
from keras.datasets import mnist
from mhgan import *

np.random.seed(1)

def mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 1: Normalize our inputs to be in the range[-1, 1]
    # 2: Expand dimensions so we have 1 channel
    x_train = np.expand_dims(
        (x_train.astype(np.float32) - 127.5)/127.5,
        3
    )
    # Return real samples
    return x_train

def create_images_from_generated(images, dimensions=(10, 10), epoch='latest'):
    plt.figure(figsize=dimensions)
    for i in range(images.shape[0]):
        plt.subplot(dimensions[0], dimensions[1], i+1)
        plt.imshow(images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('output/generated-e{}.png'.format(epoch))

def create_accumulator():
    def train_accumulator(epoch, state):
        sess, losses, [z, g, z_sampler] = state

        if epoch % 5 == 0:
            print('Generating sample images...')

            # Generate some images
            images = sess.run(g, feed_dict={
                z: z_sampler(100)
            })

            # Remove channel dimension
            images = images.squeeze(axis=3)

            # Save generated images
            create_images_from_generated(images, epoch=epoch)

        # Print current training losses
        print('Epoch {} losses G={:.6f} D={:.6f}'.format(
            epoch,
            losses.T[0][-1],
            losses.T[1][-1]
        ))
    return train_accumulator

def create_data_sampler(data):
    def sampler(batch_size):
        return data[np.random.permutation(len(data))][:batch_size]
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
    data_sampler = create_data_sampler(real_data)

    gan = WGAN(
        Generator(input_shape=noise_dimensions, output_shape=real_dimensions),
        Discriminator(),
        create_accumulator()
    )
    gan.train(
        sess,
        data_sampler,
        noise_sampler,
        batch_size=128,
        n_epochs=100,
        n_accumulate=1
    )
