from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import interface

class Model(interface.BaseModel):

    def __init__(self, params, scope="DenseModel"):
        super(Model, self).__init__(params=params, scope=scope)

    def dense_block(self, x, blocks, growth_rate, dropout, training, name, memory=None):
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
                x = self.conv_block(x, growth_rate, dropout, training, name=name + '_block' + str(i + 1), memory=memory)
        return x


    def transition_block(self, x, reduction, dropout, training, name):
        """A transition block.
        Arguments:
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.
        Returns:
            output tensor for the block.
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x = tf.layers.dropout(x, dropout, training=training)
            x = tf.layers.conv2d(x, int(x.get_shape().as_list()[-1] * reduction), 
                kernel_size=1, padding='same', use_bias=False, name='_conv')
            x = tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding="valid", name='_pool')
            x = tf.layers.batch_normalization(x, axis=-1, epsilon=1.001e-5, name='_bn')
            x = tf.nn.relu(x, name='_relu')
        return x


    def conv_block(self, x, growth_rate, dropout, training, name, memory=None):
        """A building block for a dense block.
        Arguments:
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.
        Returns:
            output tensor for the block.
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x1 = tf.layers.dropout(x, dropout, training=training)
            x1 = tf.layers.conv2d(x1, 4 * growth_rate, kernel_size=1, padding='same', use_bias=False, name='_1_conv')
            x1 = tf.layers.batch_normalization(x1, axis=-1, epsilon=1.001e-5, name='_1_bn')
            x1 = tf.nn.relu(x1, name='_1_relu')

            if memory is None:
                w_2_conv = tf.get_variable(name="w_2_conv", shape=[3, 3, x1.get_shape().as_list()[-1], growth_rate])
            else:
                w_2_conv = tf.layers.conv2d(memory, x1.get_shape().as_list()[-1] * growth_rate, kernel_size=1, 
                    padding='same', use_bias=True, name='_m_2_conv1')
                w_2_conv = tf.nn.tanh(w_2_conv, name='_m_2_act1')
                w_2_conv = tf.reshape(w_2_conv, [3, 3, x1.get_shape().as_list()[-1], growth_rate])

            x1 = tf.nn.conv2d(x1, w_2_conv, strides=[1, 1, 1, 1], padding="SAME", name='_2_conv')
            x1 = tf.layers.batch_normalization(x1, axis=-1, epsilon=1.001e-5, name='_2_bn')
            x1 = tf.nn.relu(x1, name='_2_relu')

            x = tf.concat([x, x1], axis=-1, name='_concat')
        return x


    def densenet(self, input_tensor, init_stride, blocks, growth_rate, reduction, dropout, training, memory=None):
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

        if init_stride:
            with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
                x = tf.layers.conv2d(x, 64, kernel_size=7, strides=2, padding='same', use_bias=False, name='conv')
                x = tf.layers.batch_normalization(x, axis=-1, epsilon=1.001e-5, name='bn')
                x = tf.nn.relu(x, name='relu')

            x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='same')
        else:
            with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
                x = tf.layers.conv2d(x, 64, kernel_size=3,  padding='same', use_bias=False, name='conv')
                x = tf.layers.batch_normalization(x, axis=-1, epsilon=1.001e-5, name='bn')
                x = tf.nn.relu(x, name='relu')

        for i in range(len(blocks)):
            x = self.dense_block(x, blocks[0], growth_rate, dropout, training, name='block_%s' % (i+1), memory=memory)
            if i != len(blocks) - 1:
                x = self.transition_block(x, reduction, dropout, training, name='transition_%d' % (i+1))

        x = tf.math.reduce_mean(x, axis=[1,2], name='_avg_pool')

        return x

    def model_func(self, images, lables, mode, initializer=None, regularizer=None):
        # Create model.
        params = self._params
        scope = self._scope
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, initializer=initializer, regularizer=regularizer):
            if params.dataset == "cifar10":
                class_num = params.class_num_cifar10
                init_stride = False
                blocks_num = 3
            elif params.dataset == "cifar100":
                class_num = params.class_num_cifar100
                init_stride = False
                blocks_num = 3
            elif params.dataset == "imagenet":
                class_num = params.class_num_imagenet
                init_stride = True
                blocks_num = 4
            else:
                raise ValueError("Unable to Recognize dataset: %s" % params.dataset)

            if params.use_memory:
                if params.memory_size <= 0:
                    memory_size = params.blocks_size * blocks_num * params.growth_rate
                else:
                    memory_size = params.memory_size

                memory = tf.get_variable(name="memory", shape=[1, 3, 3, memory_size])
                memory = tf.layers.dropout(memory, params.mem_drop, training=mode=="train")
            else:
                memory = None

            features = self.densenet(images, init_stride, [params.blocks_size] * blocks_num, 
                params.growth_rate, params.reduction, params.dropout, mode=="train", memory=memory)

            logits = tf.layers.dense(features, class_num, use_bias=False)

            if mode == "infer":
                return logits
            else:
                cross_loss = tf.losses.sparse_softmax_cross_entropy(lables, logits)
                if params.scale_l2 > 0.:
                    reg_loss_list = []
                    for var in tf.get_variable_scope().trainable_variables():
                        reg_loss_list.append(tf.nn.l2_loss(var))
                    reg_loss = tf.add_n(reg_loss_list, name="reg_loss") * params.scale_l2
                else:
                    reg_loss = 0.
                loss_dict = {"cross_loss":cross_loss, "reg_loss":reg_loss}

                acc_1 = tf.reduce_mean(tf.cast(tf.math.in_top_k(logits, lables, 1), tf.float32))
                acc_5 = tf.reduce_mean(tf.cast(tf.math.in_top_k(logits, lables, 5), tf.float32))
                acc_dict = {"acc_top_1":acc_1, "acc_top_5":acc_5}
                return loss_dict, acc_dict

    @staticmethod
    def get_name():
        return "DenseModel"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            # model
            blocks_size=5,
            growth_rate=12,
            class_num_cifar10=10,
            class_num_cifar100=100,
            class_num_imagenet=1000,
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
            mem_drop=0.0,
        )

        return params
