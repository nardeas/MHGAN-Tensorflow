import unittest

from mhgan import *

class Test_Topology(unittest.TestCase):

    def test_io_tensors(self):
        gan = WGAN(Generator([100], [28, 28, 1]), Discriminator())

        self.assertEqual(type(gan.G), tf.Tensor)
        self.assertEqual(type(gan.G), tf.Tensor)
        self.assertEqual(type(gan.x), tf.Tensor)
        self.assertEqual(type(gan.z), tf.Tensor)
