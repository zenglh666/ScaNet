from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import interface

class Model(interface.BaseModel):

    def __init__(self, params, scope="DenseModel"):
        super(Model, self).__init__(params=params, scope=scope)

    def transition_block(self, x, params, training, name):
        """A transition block.
        Arguments:
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.
        Returns:
            output tensor for the block.
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d(x, int(x.get_shape().as_list()[-1] * params.reduction), 
                kernel_size=1, padding='same', use_bias=False, name='_conv')
            x = tf.layers.dropout(x, params.dropout, training=training, name='_drop') 
            x = tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding="valid", name='_pool')
            x = tf.layers.batch_normalization(x, axis=-1, epsilon=1.001e-5, name='_bn')
            x = tf.nn.relu(x, name='_relu')
        return x

    def dense_block(self, x, params, blocks, training, name, memory):
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
                x = self.conv_block(x, params, training, name='_conv_block_' + str(i + 1), memory=memory)
        return x

    def conv_block(self, x, params, training, name, memory):
        """A building block for a dense block.
        Arguments:
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.
        Returns:
            output tensor for the block.
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if params.use_bc:
                x1 = tf.layers.conv2d(x, 4 * params.growth_rate, kernel_size=1, padding='same', use_bias=False, name='_1_conv')
                x1 = tf.layers.dropout(x1, params.dropout, training=training, name='_1_drop')
                x1 = tf.layers.batch_normalization(x1, axis=-1, epsilon=1.001e-5, name='_1_bn')
                x1 = tf.nn.relu(x1, name='_1_relu')
            else:
                x1 = x

            if memory is None:
                w_2_conv = tf.get_variable(name="w_2_conv", shape=[3, 3, x1.get_shape().as_list()[-1], params.growth_rate])
            else:
                w_2_conv = tf.layers.conv2d(memory, x1.get_shape().as_list()[-1] * params.growth_rate, kernel_size=1, 
                    padding='same', use_bias=False, name='_m_2_conv1')
                w_2_conv = tf.nn.tanh(w_2_conv, name='_m_2_act1')
                w_2_conv = tf.reshape(w_2_conv, [3, 3, x1.get_shape().as_list()[-1], params.growth_rate])

            x1 = tf.nn.conv2d(x1, w_2_conv, strides=[1, 1, 1, 1], padding="SAME", name='_2_conv')
            x1 = tf.layers.dropout(x1, params.dropout, training=training, name='_2_drop')
            x1 = tf.layers.batch_normalization(x1, axis=-1, epsilon=1.001e-5, name='_2_bn')
            x1 = tf.nn.relu(x1, name='_2_relu')
            x = tf.concat([x, x1], axis=-1, name='_concat')
        return x


    def model(self, x, params, training, memory=None):
        """Instantiates the DenseNet architecture.
        Arguments:
            blocks: numbers of building blocks for the four dense layers.
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, must to be specified
        Returns:
            A model instance.
        """
        with tf.variable_scope("_densenet", reuse=tf.AUTO_REUSE):
            if params.blocks_size > 0 and params.blocks_num > 0:
                blocks_size = [params.blocks_size] * params.blocks_num
            elif params.net_name is not None:
                pass
            else:
                raise ValueError("Unable to Recognize Block Size")

            if params.dataset == "cifar10" or params.dataset == "cifar10":
                with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
                    x = tf.layers.conv2d(x, 2 * params.growth_rate, kernel_size=3,  padding='same', use_bias=False, name='_conv')
                    x = tf.layers.batch_normalization(x, axis=-1, epsilon=1.001e-5, name='_bn')
                    x = tf.nn.relu(x, name='_relu')
            else:
                with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
                    x = tf.layers.conv2d(x, 64, kernel_size=7, strides=2, padding='same', use_bias=False, name='_conv')
                    x = tf.layers.batch_normalization(x, axis=-1, epsilon=1.001e-5, name='_bn')
                    x = tf.nn.relu(x, name='_relu')
                x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='same', name='_max_pool')
                
            for i in range(len(blocks_size)):
                x = self.dense_block(x, params, blocks_size[i], training, name='block_%s' % (i+1), memory=memory)
                if i != len(blocks_size) - 1:
                    x = self.transition_block(x, params, training, name='transition_%d' % (i+1))

            x = tf.math.reduce_mean(x, axis=[1,2], name='_avg_pool')

        return x

    @staticmethod
    def get_name():
        return "DenseModel"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            # model
            use_bc=False,
            blocks_num=3,
            blocks_size=13,
            growth_rate=12,
            net_name=None,
            reduction=0.5,
            batch_size=256,
            scale_l1=0.0,
            scale_l2=0.0001,
            train_steps=60000,
            decay_steps=20000,
            eval_steps=2000,
            dropout=0.0,
            use_memory=False,
            memory_size=0,
            max_memory_size=8192,
            mem_drop=0.0,
        )

        return params
