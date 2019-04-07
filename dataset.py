# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf
import sys
import os 

def get_train_eval_input(mode, params):
    if params.dataset == "cifar10" or params.dataset == "cifar100":
        train_num = 50000
        val_num = 10000
        num = train_num if mode == "train" else val_num
        return lambda:cifar(mode, params), num
    else:
        raise ValueError('unable to recognize dataset %s' % params.dataset)


def preprocess_image_cifar(image, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
            image, 40, 40)

        # Randomly crop a [HEIGHT, WIDTH] section of the image.
        image = tf.image.random_crop(image, [32, 32, 3])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image

def cifar(mode, params):
    image_bytes = 32 * 32 * 3
    train_num = 50000
    val_num = 10000
    if params.dataset == "cifar10":
        filename_train = ["data_batch_1.bin","data_batch_2.bin",
            "data_batch_3.bin","data_batch_4.bin","data_batch_5.bin"]
        filename_test = ["test_batch.bin"]
        label_bytes = 1
    elif params.dataset == "cifar100":
        filename_train = ["train.bin"]
        filename_test = ["test.bin"]
        label_bytes = 2
    else:
        raise ValueError('unable to handle dataset %s' % params.dataset)

    data_files = []
    if mode == "train":
        for f in filename_train:
            data_files.append(os.path.join(params.input, f))
    else:
        for f in filename_test:
            data_files.append(os.path.join(params.input, f))

    for f in data_files:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.device("/cpu:0"):       
        record_bytes = label_bytes + image_bytes
        dataset = tf.data.FixedLengthRecordDataset(data_files, record_bytes)

        dataset = dataset.map(
            lambda example_serialized: tf.decode_raw(example_serialized, tf.uint8)
        )

        dataset = dataset.map(
            lambda record: (
                tf.transpose(
                    tf.reshape(
                        tf.slice(record, [label_bytes], [image_bytes]),
                        [3, 32, 32]
                    ),
                    [1, 2, 0]
                ),
                tf.reshape(
                    tf.cast(
                        tf.slice(record, [label_bytes - 1], [label_bytes]), 
                        tf.int32), 
                    []
                )
            )
        )

        
        if mode == "train":
            dataset = dataset.cache()
            dataset = dataset.repeat()
            dataset = dataset.shuffle(params.buffer_size)

        dataset = dataset.map(
            lambda image, label: (
                preprocess_image_cifar(image, mode=="train"),
                label
            )
        )

        # Create iterator
        dataset = dataset.batch(params.batch_size)
        dataset = dataset.prefetch(params.pre_fetch)
    return dataset

