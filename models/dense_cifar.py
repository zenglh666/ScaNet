from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import interface

class Model(interface.DenseModel):

    def __init__(self, params, scope="DenseCifarModel"):
        super(Model, self).__init__(params=params, scope=scope)

    def model_func(self, images, lables, mode, initializer=None, regularizer=None):
        # Create model.
        params = self._params
        scope = self._scope
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, initializer=initializer, regularizer=regularizer):
            features = self.densenet(False, [params.blocks_size for b in params.blocks_num], params.growth_rate, images, params.reduction)

            if params.dataset == "cifar10":
                class_num = params.class_num_cifar10
            elif params.dataset == "cifar100":
                class_num = params.class_num_cifar100
            logits = tf.layers.dense(features, class_num, use_bias=False)

            if mode == "infer":
                return logits
            else:
                cross_loss = tf.losses.sparse_softmax_cross_entropy(lables, logits)
                if params.scale_l2 > 0.:
                    reg_loss_list = []
                    for var in tf.trainable_variables():
                        reg_loss_list.append(tf.nn.l2_loss(var))
                    reg_loss = params.scale_l2 * tf.add_n(reg_loss_list, name="reg_loss")
                else:
                    reg_loss = 0.
                loss_dict = {"cross_loss":cross_loss, "reg_loss":reg_loss}

                acc_1 = tf.reduce_mean(tf.cast(tf.math.in_top_k(logits, lables, 1), tf.float32))
                acc_5 = tf.reduce_mean(tf.cast(tf.math.in_top_k(logits, lables, 5), tf.float32))
                acc_dict = {"acc_top_1":acc_1, "acc_top_5":acc_5}
                return loss_dict, acc_dict

    @staticmethod
    def get_name():
        return "DenseCifarModel"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            # model
            blocks_num = 3,
            blocks_size = 11,
            growth_rate=12,
            class_num_cifar10 = 10,
            class_num_cifar100 = 100,
            reduction = 0.5,
            batch_size=256,
            scale_l1=0.0,
            scale_l2=1e-4,
            train_steps=90000,
            decay_steps=30000,
            eval_steps=3000,
        )

        return params
