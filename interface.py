from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class BaseModel(object):
    """ Abstract object representing an NMT model """

    def __init__(self, params, scope):
        self._scope = scope
        self._params = params

    def model(self, x, params, training, memory=None):
        """
        :param initializer: the initializer used to initialize the model
        :param regularizer: the regularizer used for model regularization
        """
        raise NotImplementedError("Not implemented")

    def model_func(self, images, lables, mode, params=None, initializer=None, regularizer=None):
        # Create model.
        params = params or self._params
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE, initializer=initializer, regularizer=regularizer) as scope:
            if params.use_memory:
                if params.memory_size <= 0:
                    memory_size = params.blocks_size * params.blocks_num * params.growth_rate
                    memory_size = min(memory_size, params.max_memory_size)
                else:
                    memory_size = params.memory_size

                memory = tf.get_variable(name="memory", shape=[1, 3, 3, memory_size])
                memory = tf.layers.dropout(memory, params.mem_drop, training=mode=="train")
            else:
                memory = None

            features = self.model(images, params, mode=="train", memory=memory)
            logits = tf.layers.dense(features, params.class_num, use_bias=False)

            if mode == "infer":
                return logits
            else:
                cross_loss = tf.losses.sparse_softmax_cross_entropy(lables, logits)
                if params.scale_l2 > 0.:
                    reg_loss_list = []
                    ignore_list = []
                    for var in scope.trainable_variables():
                        if 'gamma' not in var.name:
                            reg_loss_list.append(tf.nn.l2_loss(var))
                        else:
                            reg_loss_list.append(tf.nn.l2_loss(var - 1))
                            ignore_list.append(var)
                    reg_loss = tf.add_n(reg_loss_list, name="_reg_loss") * params.scale_l2
                else:
                    reg_loss = 0.
                loss_dict = {"cross_loss":cross_loss, "reg_loss":reg_loss}

                acc_1 = tf.reduce_mean(tf.cast(tf.math.in_top_k(logits, lables, 1), tf.float32))
                acc_5 = tf.reduce_mean(tf.cast(tf.math.in_top_k(logits, lables, 5), tf.float32))
                acc_dict = {"acc_top_1":acc_1, "acc_top_5":acc_5}

                return loss_dict, acc_dict

    @staticmethod
    def get_name():
        raise NotImplementedError("Not implemented")

    @staticmethod
    def get_parameters(dataset=None):
        raise NotImplementedError("Not implemented")

    @property
    def parameters(self):
        return self._params
