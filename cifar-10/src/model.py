import tensorflow as tf
import numpy as np


SAME_PADDING = "SAME"


class ConvoModel(object):

    def __init__(self, config):

        self._config = config

    def initialize(self, x, keep_prob):

        # first convolution layer: output_shape (batch_size, 16, 16, 64)
        self._convo_1 = self._get_convo_layer(
            x,
            a_function=self._config.a_function,
            size=5,
            in_channels=3,
            out_channels=64,
            stride=1,
            stddev=np.sqrt(2.0/75.0),
            name="convo_1"
        )

        tf.histogram_summary("conv_1/activations", self._convo_1)

        self._max_switches_1, self._max_pool_1 = self._get_max_pool_layer(
            self._convo_1,
            size=3,
            stride=2,
            name="max_pool_1"
        )

        """
        self._norm_1 = tf.nn.lrn(
            self._max_pool_1,
            4,
            bias=1.0,
            alpha=0.001 / 9.0,
            beta=0.75,
            name="norm_1"
        )
        """

        # second convolution layer: output_shape (batch_size, 8, 8, 64)
        self._convo_2 = self._get_convo_layer(
            self._max_pool_1,
            a_function=self._config.a_function,
            size=2,
            in_channels=64,
            out_channels=64,
            stride=1,
            stddev=np.sqrt(2.0/256.0),
            name="convo_2"
        )

        tf.histogram_summary("conv_2/activations", self._convo_2)

        """
        self._norm_2 = tf.nn.lrn(
            self._convo_2,
            4,
            bias=1.0,
            alpha=0.001 / 9.0,
            beta=0.75,
            name="norm_2"
        )
        """

        self._max_switches_2, self._max_pool_2 = self._get_max_pool_layer(
            self._convo_2,
            size=2,
            stride=2,
            name="max_pool_2"
        )

        # densely-connected network
        self._dense_layer_1 = self._get_dense_layer(
            tf.reshape(self._max_pool_2, (100, -1)),
            input_size=8*8*64,
            hidden_units=2048,
            a_function=self._config.a_function,
            stddev=np.sqrt(2.0/4096.0),
            name="dense_layer_1"
        )

        tf.histogram_summary("dense_1/activations", self._dense_layer_1)

        self._dense_layer_2 = self._get_dense_layer(
            self._dense_layer_1,
            input_size=2048,
            hidden_units=1024,
            a_function=self._config.a_function,
            stddev=np.sqrt(2.0/2048.0),
            name="dense_layer_2"
        )

        tf.histogram_summary("dense_2/activations", self._dense_layer_2)

        # softmax layer
        self._softmax_logit, self._softmax_layer = self._get_softmax_layer(
            self._dense_layer_2,
            input_size=1024,
            output_size=10,
            name="softmax"
        )

    def train_op(self, y, global_step):

        self._softmax_loss_layer = tf.nn.sparse_softmax_cross_entropy_with_logits(
            self._softmax_logit,
            y
        )

        initial_learning_rate = 0.1
        num_batches_per_epoch = self._config.examples_per_epoches / self._config.batch_size
        decay_steps = int(num_batches_per_epoch * 350.0)

        lr = tf.train.exponential_decay(
            initial_learning_rate,
            global_step,
            decay_steps,
            0.1,
            staircase=True
        )

        optimizer = tf.train.GradientDescentOptimizer(0.1)

        cross_entropy_mean = tf.reduce_mean(self._softmax_loss_layer)
        tf.add_to_collection("losses", cross_entropy_mean)

        loss = tf.add_n(tf.get_collection("losses"), name="loss")

        tf.scalar_summary("loss", loss)

        return loss, optimizer.minimize(loss, global_step)

    def infer(self):

        return self._softmax_layer

    def _get_variable_with_summary(self, initializer, shape, wd=None, name=""):

        variable = tf.get_variable(
            name,
            shape=shape,
            initializer=initializer
        )

        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(variable), wd, name="weight_loss")
            tf.add_to_collection("losses", weight_decay)

        tf.histogram_summary("%s" % name, variable)
        tf.scalar_summary("%s/sparsity" % name, tf.nn.zero_fraction(variable))

        return variable

    def _get_convo_layer(
            self,
            input,
            a_function,
            size,
            in_channels,
            out_channels,
            stride,
            padding_config=SAME_PADDING,
            stddev=5e-2,
            name=""):

        weights = self._get_variable_with_summary(
            initializer=tf.truncated_normal_initializer(
                stddev=stddev,
                dtype=tf.float32
            ),
            shape=(size, size, in_channels, out_channels),
            name="%s_weights" % name,
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

        max_pool = tf.nn.max_pool(
            input,
            ksize=[1, size, size, 1],
            strides=[1, stride, stride, 1],
            padding=padding_config,
            name=name
        )

        _, switches = tf.nn.max_pool_with_argmax(
            input,
            ksize=[1, size, size, 1],
            strides=[1, stride, stride, 1],
            padding=padding_config,
            name=name
        )

        return (switches, max_pool)

    def _get_dense_layer(
        self,
        input,
        input_size,
        hidden_units,
        a_function,
        stddev=5e-2,
        name=""
    ):

        weights = self._get_variable_with_summary(
            initializer=tf.truncated_normal_initializer(
                stddev=stddev,
                dtype=tf.float32
            ),
            wd=0.004,
            shape=(input_size, hidden_units),
            name="%s_weights" % name,
        )

        biases = tf.get_variable(
            "%s_biases" % name,
            shape=(hidden_units),
            initializer=tf.constant_initializer(0.0)
        )

        return a_function(tf.matmul(input, weights) + biases)

    def _get_softmax_layer(
        self,
        input,
        input_size,
        output_size,
        name=""
    ):

        weights = self._get_variable_with_summary(
            initializer=tf.truncated_normal_initializer(
                    stddev=np.sqrt(1.0/1024),
                    dtype=tf.float32
            ),
            shape=(input_size, output_size),
            name="%s_weights" % name
        )

        biases = tf.get_variable(
            "%s_biases" % name,
            shape=(output_size),
            initializer=tf.constant_initializer(0.0)
        )

        logits = tf.matmul(input, weights) + biases

        return logits, tf.nn.softmax(logits)
