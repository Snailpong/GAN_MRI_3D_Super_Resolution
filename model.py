import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
from losses import VGGLoss

from layers import (ResidualBlock, ConvBlock)


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1, reduction='none')


def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(cross_entropy(tf.ones_like(real_output), real_output))
    fake_loss = tf.reduce_mean(cross_entropy(tf.zeros_like(fake_output), fake_output))
    total_loss = real_loss + fake_loss

    return total_loss


def generator_loss(batch_hr, batch_es, fake_output):
    cross_entropy_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    mse_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(batch_hr, batch_es))

    return mse_loss + 1e-3 * cross_entropy_loss


class Generator(Model):
    def __init__(self, num_residual_blocks=4):
        super(Generator, self).__init__()
        self.num_residual_blocks = num_residual_blocks

        self.conv_1 = tf.keras.layers.Conv3D(16, 9, 1, 'same')
        self.conv_2 = tf.keras.layers.Conv3D(16, 3, 1, 'same')
        self.conv_3 = tf.keras.layers.Conv3D(1, 9, 1, 'same')
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1, 2, 3])
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.8)

        self.residual_blocks = tf.keras.Sequential(
            [ResidualBlock(16) for _ in range(num_residual_blocks)],
            name='ResidualBlocks')

    def call(self, image, training=False):
        x = self.conv_1(image)
        x = self.prelu(x)
        skip = x
        x = self.residual_blocks(x, training=training)
        x = self.conv_2(x)
        x = self.bn(x, training=training)
        x = x + skip
        x = self.conv_3(x)
        sr_image = tf.nn.tanh(x)
        return sr_image


class Discriminator(tf.keras.Model):

    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.conv_1 = tf.keras.layers.Conv3D(8, 3, 1, 'same')
        self.conv_blocks = tf.keras.Sequential(name='ConvBlocks')
        self.conv_blocks.add(ConvBlock(16, 2))
        for i in range(1, 5):
            self.conv_blocks.add(ConvBlock(16 * 2**i, 1))
            self.conv_blocks.add(ConvBlock(16 * 2**i, 2))
        self.fc_1 = tf.keras.layers.Dense(1024)
        self.fc_2 = tf.keras.layers.Dense(1)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, image, training=None):
        x = self.conv_1(image)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.conv_blocks(x, training=training)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.fc_2(x)
        # x = tf.math.sigmoid(x)
        return x
