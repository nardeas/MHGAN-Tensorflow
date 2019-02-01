from keras.datasets import mnist
from helpers import *
from mhgan import *

np.random.seed(1)

def mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 1: Normalize our inputs to be in the range[-1, 1]
    # 2: Expand dimensions so we have 1 channel
    return np.expand_dims((x_train.astype(np.float32) - 127.5)/127.5, 3)

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

    print('Training WGAN on MNIST data...')
    gan = WGAN(
        Generator(input_shape=noise_dimensions, output_shape=real_dimensions),
        Discriminator()
    )
    gan.train_helper = create_train_helper(
        sess=sess,
        sample_count=25,
        sample_nth=10,
        sample_save=True,
        output_path='output/summaries/wgan',
    )
    gan.train(
        sess,
        data_sampler,
        noise_sampler,
        batch_size=32,
        n_epochs=100,
        n_accumulate=1
    )
