import tensorflow as tf


SAME_PADDING = "SAME"


class ConvoModel(object):

    def __init__(self, x, y, config):

        self._config = config

        self._x = x
        self._y = y

    def initialize(self):

        # first convolution layer: output_shape (batch_size, 16, 16, 64)
        self._convo_1 = self._get_convo_layer(
            self._x,
            a_function=tf.sigmoid,
            size=5,
            in_channels=3,
            out_channels=64,
            stride=1,
            name="convo_1"
        ) 

        self._max_pool_1 = self._get_max_pool_layer(
            self._convo_1,
            size=3,
            stride=2,
            name="max_pool_1"
        )

        # second convolution layer: output_shape (batch_size, 8, 8, 64)
        self._convo_2 = self._get_convo_layer(
            self._max_pool_1,
            a_function=tf.sigmoid,
            size=2,
            in_channels=64,
            out_channels=64,
            stride=1,
            name="convo_2"
        ) 

        self._max_pool_2 = self._get_max_pool_layer(
            self._convo_2,
            size=2,
            stride=2,
            name="max_pool_2"
        )

    def _get_convo_layer(
        self,
        input,
        a_function,
        size,
        in_channels,
        out_channels,
        stride,
        padding_config=SAME_PADDING,
        name=""):

        weights = tf.get_variable(
            "%s_weights" % name, 
            shape=(size, size, in_channels, out_channels),
            initializer=tf.random_uniform_initializer(-.00001, 0.00001)
        )

        biases = tf.get_variable(
            "%s_biases" % name, 
            shape=(out_channels),
            initializer=tf.constant_initializer(0.0)
        )

        convo = tf.nn.conv2d(
            input,
            weights,
            strides=(1, stride, stride, 1),
            padding=padding_config,
            name=name
        )

        convo = tf.nn.bias_add(convo, biases)

        return a_function(convo) 

    def _get_max_pool_layer(
        self,
        input,
        size,
        stride,
        padding_config=SAME_PADDING,
        name=""
    ):

        return tf.nn.max_pool(
            input,
            ksize=[1, size, size, 1],
            strides=[1, stride, stride, 1],
            padding=padding_config,
            name=name
        )
