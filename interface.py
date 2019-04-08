from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class DenseModel(object):
    """ Abstract object representing an NMT model """

    def __init__(self, params, scope):
        self._scope = scope
        self._params = params

    def dense_block(self, x, blocks, growth_rate, name):
        """A dense block.
        Arguments:
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.
        Returns:
            output tensor for the block.
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for i in range(blocks):
                x = self.conv_block(x, growth_rate, name=name + '_block' + str(i + 1))
        return x


    def transition_block(self, x, reduction, name):
        """A transition block.
        Arguments:
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.
        Returns:
            output tensor for the block.
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x = tf.nn.relu(x, name='_relu')
            x = tf.layers.conv2d(x, int(x.get_shape().as_list()[-1] * reduction), kernel_size=1, padding='same', use_bias=False, name='_conv')
            x = tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding="same", name='_avg_pool')
            x = tf.layers.batch_normalization(x, axis=-1, epsilon=1.001e-5, name='_bn')
        return x


    def conv_block(self, x, growth_rate, name):
        """A building block for a dense block.
        Arguments:
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.
        Returns:
            output tensor for the block.
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x1 = tf.nn.relu(x, name='_0_relu')
            x1 = tf.layers.conv2d(x, 4 * growth_rate, kernel_size=1, padding='same', use_bias=False, name='_1_conv')

            x1 = tf.layers.batch_normalization(x1, axis=-1, epsilon=1.001e-5, name='_1_bn')
            x1 = tf.nn.relu(x1, name='_1_relu')
            w_2_conv = tf.get_variable(name="w_2_conv", shape=[3, 3, x1.get_shape().as_list()[-1], growth_rate])
            x1 = tf.nn.conv2d(x1, w_2_conv, strides=[1, 1, 1, 1], padding="SAME", name='_2_conv')
            x1 = tf.layers.batch_normalization(x1, axis=-1, epsilon=1.001e-5, name='_2_bn')

            x = tf.concat([x, x1], axis=-1, name='_concat')
        return x


    def densenet(self, init_conv, blocks, growth_rate, input_tensor, reduction=0.5):
        """Instantiates the DenseNet architecture.
        Arguments:
            blocks: numbers of building blocks for the four dense layers.
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, must to be specified
        Returns:
            A model instance.
        """

        x = input_tensor

        if init_conv:
            with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
                x = tf.layers.conv2d(x, 64, kernel_size=7, strides=2, padding='same', use_bias=False, name='conv')
                x = tf.layers.batch_normalization(x, axis=-1, epsilon=1.001e-5, name='bn')
                x = tf.nn.relu(x, name='relu')

            x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='same')

        for i in range(len(blocks)):
            x = self.dense_block(x, blocks[0], growth_rate, name='block_%s' % (i+1))
            if i != len(blocks) - 1:
                x = self.transition_block(x, reduction, name='transition_%d' % (i+1))

        x = tf.layers.batch_normalization(x, axis=-1, epsilon=1.001e-5, name='bn')
        x = tf.math.reduce_mean(x, axis=[1,2], name='_avg_pool')

        return x


    def model_func(self, images, lables, mode, initializer=None, regularizer=None):
        """
        :param initializer: the initializer used to initialize the model
        :param regularizer: the regularizer used for model regularization
        """
        raise NotImplementedError("Not implemented")

    @staticmethod
    def get_name():
        raise NotImplementedError("Not implemented")

    @staticmethod
    def get_parameters():
        raise NotImplementedError("Not implemented")

    @property
    def parameters(self):
        return self._params
