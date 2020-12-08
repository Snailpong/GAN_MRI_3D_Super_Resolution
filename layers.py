import tensorflow as tf



class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.conv_1 = tf.keras.layers.Conv3D(filters, 3, 1, 'same')
        self.conv_2 = tf.keras.layers.Conv3D(filters, 3, 1, 'same')
        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=0.8)
        self.bn_2 = tf.keras.layers.BatchNormalization(momentum=0.8)
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1, 2, 3])

    def call(self, x, training=None):
        skip = x
        x = self.conv_1(x)
        x = self.bn_1(x, training=training)
        x = self.prelu(x)
        x = self.conv_2(x)
        x = self.bn_2(x, training=training)
        return x + skip

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return [
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            self.filters,
        ]

    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({'filters': self.filters})
        return config


class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, filters, stride, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.conv = tf.keras.layers.Conv3D(filters, 3, stride, 'same')
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.8)

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return tf.nn.leaky_relu(x, alpha=0.2)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return [
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            self.filters,
        ]

    def get_config(self):
        config = super(ConvBlock, self).get_config()
        config.update({'filters': self.filters})
        return config
