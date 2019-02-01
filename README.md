# Metropolis-Hastings GAN (MHGAN)

MHGAN implemented in Tensorflow (mostly) as described in the original paper:

https://arxiv.org/pdf/1811.11357.pdf

## Overview

The base network is a WGAN with DCGAN generator and discriminator. As opposed to the standard DCGAN we are using GELU activation as this is shown to generally improve performance:

https://arxiv.org/pdf/1606.08415.pdf

Metropolis-Hastings GAN refers to the functionality of improving trained GANs by drawing **k** samples from the generator in MCMC fashion and using the discriminator (or critic) probabilities for calculating an acceptance ratio to obtain the best possible sample. The original paper argues that given perfect discriminator, and **k** approaching infinity, we can obtain samples from the true data distribution.

Thus, even if the generator doesn't converge optimally, we can use the discriminator to draw enhanced samples from the network.

The `mhgan.py` module provides a wrapper for a trained generator/discriminator pair with utility methods to draw better samples. The chain is calibrated using a score from real data as starting point to avoid burn-in periods.

## Experiments

**Training metrics:**

<img src="/media/g_loss.png" height="150" />
<img src="/media/d_loss.png" height="150" />

<img src="/media/g_loss_dist.png" height="150" />
<img src="/media/d_loss_dist.png" height="150" />

**Convergence on MNIST subset:**

 <img src="/media/gan.gif" width="300" />

**Basic sample after 1500 epochs:**

![Sample](/media/sample.png?raw=true "Basic sample")

**Enhanced sample after 1500 epochs:**

![MH-Sample](/media/mh_sample.png?raw=true "MH sample")

## Notes

Check the `test_mnist.ipynb` notebook for examples. The basic flow is this:

**Train a (W)GAN:**
```python
gan = WGAN(
    Generator(
      input_shape=noise_dimensions,
      output_shape=real_dimensions
    ),
    Discriminator()
)
gan.train(
    sess,
    data_sampler,
    noise_sampler,
    batch_size=32,
    n_epochs=100,
    n_accumulate=1
)
```

**Wrap the GAN in MHGAN instance and draw enhanced samples:**
```python
mhgan = MHGAN(gan)
mhgan.generate_enhanced(
    sess,
    data_sampler,
    noise_sampler,
    count=16,
    k=1000
)
```

## Future

Experiment with weight normalization vs. batch normalization:

https://arxiv.org/pdf/1704.03971.pdf
