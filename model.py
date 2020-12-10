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
    #cross_entropy_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    mse_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(batch_hr, batch_es))

    #return mse_loss + 1e-3 * cross_entropy_loss
    return mse_loss


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
        self.conv_1 = tf.keras.layers.Conv3D(16, 3, 1, 'same')
        self.conv_blocks = tf.keras.Sequential(name='ConvBlocks')
        self.conv_blocks.add(ConvBlock(16, 2))
        for i in range(1, 3):
            self.conv_blocks.add(ConvBlock(4 * 2**i, 1))
            self.conv_blocks.add(ConvBlock(4 * 2**i, 2))
        self.fc_1 = tf.keras.layers.Dense(16)
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
        x = tf.math.sigmoid(x)
        return x


class SRGan(tf.keras.Model):

    def __init__(self, upscale_factor=4, generator_weights=None, **kwargs):
        super(SRGan, self).__init__(**kwargs)
        self.generator = Generator(weights=generator_weights)
        self.discriminator = Discriminator()
        self.vgg_loss = VGGLoss()


    def compile(self, d_optimizer, g_optimizer, loss_fn, **kwargs):
        super(SRGan, self).compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        lr_images, hr_images = data
        batch_size = tf.shape(lr_images)[0]

        ones = tf.ones([batch_size])
        zeros = tf.zeros([batch_size])

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            sr_images = self.generator(lr_images, training=True)

            fake_logits = self.discriminator(sr_images, training=True)
            real_logits = self.discriminator(hr_images, training=True)

            d_loss_fake = tf.reduce_mean(self.loss_fn(zeros, fake_logits))
            d_loss_real = tf.reduce_mean(self.loss_fn(ones, real_logits))
            d_loss = d_loss_fake + d_loss_real

            content_loss = self.vgg_loss(hr_images, sr_images)
            g_loss = tf.reduce_mean(self.loss_fn(ones, fake_logits))
            perceptual_loss = content_loss + 1e-3 * g_loss

            d_loss_scaled = \
                d_loss / self.distribute_strategy.num_replicas_in_sync
            perceptual_loss_scaled = \
                perceptual_loss / self.distribute_strategy.num_replicas_in_sync

        d_grads = d_tape.gradient(d_loss_scaled,
                                  self.discriminator.trainable_weights)
        g_grads = g_tape.gradient(perceptual_loss_scaled,
                                  self.generator.trainable_weights)

        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_weights))
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_weights))

        return {
            'perceptual_loss': perceptual_loss,
            'content_loss': content_loss,
            'g_loss': g_loss,
            'd_loss_real': d_loss_real,
            'd_loss_fake': d_loss_fake
        }