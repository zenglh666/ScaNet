from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf
import dataset
import models
from tensorflow.python.client import device_lib


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training scalble cnn models",
        usage="trainer.py [<args>] [-h | --help]"
    )
    parser.add_argument("--job_id", type=str, default="",
                        help="id of job")

    # input files
    parser.add_argument("--output", type=str, default="",
                        help="Path to saved models")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to pre-trained checkpoint")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")
    parser.add_argument("--dataset", type=str, required=True,
                        help="dataset to use")

    return parser.parse_args(args)


def default_parameters():
    params = tf.contrib.training.HParams(
        # Default Path and Name
        dataset="",
        data_dir="/data/zenglh/data",
        input1="cifar/cifar10/cifar-10-batches-bin",
        input2="cifar/cifar100/cifar-100-binary", 
        input3="imagenet_scale256",
        input="",
        output="/data/zenglh/ScaNet/results",
        model="",
        job_id = "",
        log_id= "",
        restore_memory_file="",
        # Default hyper parameters
        pre_fetch=8,
        buffer_size=8192,
        log_steps=100,
        class_num=None,
        class_num_cifar10=10,
        class_num_cifar100=100,
        class_num_imagenet=1000,
        # Default training hyper parameters
        initializer="normal_unit_scaling",
        initializer_gain=1.0,
        initial_mode='fan_in',
        batch_size=256,
        gpu_num=0,
        scale_l1=0.0,
        scale_l2=0.0,
        # Default Optimier hyper parameters
        optimizer="Mom",
        use_nesterov=False,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        clip_grad_norm=1.,
        # Default Learning rate hyper parameters
        learning_rate=1e-1,
        learning_rate_decay="exponential_decay",
        train_steps=300000,
        decay_steps=100000,
        # Default Check Point hyper parameters
        keep_checkpoint_max=1,
        keep_top_checkpoint_max=1,
        summary_steps=1e10,
        # Default Validation hyper parameters
        eval_steps=10000,
        infer_in_validation=False,
        # Default Memory Parameter
        use_memory=False,
        memory_size=0,
        memory_train=False,
    )
    return params

def export_params(output_dir, name, params):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)
    with tf.gfile.Open(filename, "w") as fd:
        fd.write(params.to_json())

def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in params1.values().items():
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in params2.values().items():
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def override_parameters(params, args):
    params.model = args.model
    params.dataset = args.dataset
    params.output = args.output or params.output
    params.job_id = args.job_id or params.job_id
    params.parse(args.parameters)
    timestr =  datetime.now().isoformat().replace(':','-').replace('.','MS')
    if params.log_id == "":
        params.log_id = timestr
    if params.job_id == "":
        params.job_id = timestr
    params.output = os.path.join(params.output, params.job_id)

    if params.dataset == "cifar10":
        params.input = os.path.join(params.data_dir, params.input1)
        params.class_num = params.class_num or params.class_num_cifar10
    elif params.dataset == "cifar100":
        params.input = os.path.join(params.data_dir, params.input2)
        params.class_num = params.class_num or params.class_num_cifar100
    elif params.dataset == "imagenet":
        params.input = os.path.join(params.data_dir, params.input3)
        params.class_num = params.class_num or params.class_num_imagenet
    else:
        raise ValueError("Unrecognized dataset: %s" % params.dataset)

    if params.gpu_num == 0:
        local_device_protos = device_lib.list_local_devices()
        for x in local_device_protos:
            if x.device_type == 'GPU':
                params.gpu_num += 1

    return params


def get_initializer(params):
    if params.initializer == "uniform":
        max_val = params.initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val)
    elif params.initializer == "normal":
        return tf.random_normal_initializer(0.0, params.initializer_gain)
    elif params.initializer == "normal_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain, mode=params.initial_mode,
                                               distribution="truncated_normal")
    elif params.initializer == "uniform_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain, mode=params.initial_mode,
                                               distribution="uniform")
    else:
        raise ValueError("Unrecognized initializer: %s" % params.initializer)


def get_learning_rate_decay(global_step, params):
    if params.learning_rate_decay == "exponential_decay":
        return tf.train.exponential_decay(params.learning_rate, global_step, 
            params.decay_steps, decay_rate=0.1, staircase=True)
    elif params.learning_rate_decay == "none":
        return tf.convert_to_tensor(params.learning_rate, dtype=tf.float32)
    else:
        raise ValueError("Unrecognized learning_rate_decay: %s" % params.learning_rate_decay)


def get_optimizer(learning_rate, params):
    if params.optimizer == "Adam":
        opt = tf.train.AdamOptimizer(learning_rate,
                                     beta1=params.adam_beta1,
                                     beta2=params.adam_beta2,
                                     epsilon=params.adam_epsilon)
    elif params.optimizer == "LazyAdam":
        opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate,
                                               beta1=params.adam_beta1,
                                               beta2=params.adam_beta2,
                                               epsilon=params.adam_epsilon)
    elif params.optimizer == "Mom":
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=params.use_nesterov)
    elif params.optimizer == "Sgd":
        opt = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError("Unrecognized Optimizer: %s" % params.optimizer)
    return opt

def run_config(params):
    optimizer_options = tf.OptimizerOptions(
        opt_level=tf.OptimizerOptions.L1,
        do_function_inlining=True,
    )
    graph_options = tf.GraphOptions(
        optimizer_options=optimizer_options,
        place_pruned_graph=True,
        enable_bfloat16_sendrecv=False,
        build_cost_model=0
    )

    gpu_options=tf.GPUOptions(allow_growth=True)

    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        graph_options=graph_options,
        gpu_options=gpu_options)

    mirrored_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=params.gpu_num)

    run_config = tf.estimator.RunConfig(
        model_dir=params.output,
        save_summary_steps=params.summary_steps,
        save_checkpoints_steps=params.eval_steps,
        keep_checkpoint_max=1,
        log_step_count_steps=params.log_steps,
        session_config=session_config,
        train_distribute=mirrored_strategy,
        eval_distribute=mirrored_strategy
    )
    return run_config

def print_parameters():
    all_weights = {v.name: v for v in tf.trainable_variables()}
    total_size = 0

    for v_name in sorted(list(all_weights)):
        v = all_weights[v_name]
        tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                        str(v.shape).ljust(20))
        v_size = np.prod(np.array(v.shape.as_list())).tolist()
        total_size += v_size
    tf.logging.info("Total trainable variables size: %d", total_size)

def model_fn(features, labels, mode, params):
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        images = features
        # Create model
        model_cls = models.get_model(params.model)
        model = model_cls(params)

        if mode == tf.estimator.ModeKeys.TRAIN:
            initializer = get_initializer(params)
            losses_dict, acc_dict = model.model_func(images, labels, mode="train", initializer=initializer)

            # Create global step
            global_step = tf.train.get_or_create_global_step()

            # Create optimizer
            learning_rate = get_learning_rate_decay(global_step, params)
            opt = get_optimizer(learning_rate, params)
            if params.clip_grad_norm > 0.:
                opt = tf.contrib.estimator.clip_gradients_by_norm(opt, params.clip_grad_norm)

            loss = tf.add_n([v for v in losses_dict.values()], name="loss")
            ops = opt.minimize(loss, global_step)

            logging_dict = {}
            logging_dict.update(losses_dict)
            logging_dict.update(acc_dict)

            with tf.device("/cpu:0"):
                for k,v in logging_dict.items():
                    tf.summary.scalar(k, v)

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=ops)
        elif mode == tf.estimator.ModeKeys.EVAL:
            losses_dict, acc_dict = model.model_func(images, labels, mode="eval")
            loss = tf.add_n([v for v in losses_dict.values()], name="loss")

            metric_dict = {}
            for k, v in losses_dict.items():
                metric_dict[k] = tf.metrics.mean(v)
            for k, v in acc_dict.items():
                metric_dict[k] = tf.metrics.mean(v)

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metric_dict)
        else:
            logits = model.model_func(images, labels, mode="infer")
            return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    model_cls = models.get_model(args.model)
    params = default_parameters()

    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = merge_parameters(params, model_cls.get_parameters(args.dataset))
    override_parameters(params, args)

    # Export all parameters and model specific parameters
    export_params(params.output, "params.json", params)

    log = logging.getLogger('tensorflow')
    formatter = logging.Formatter('%(asctime)s - %(name)s:%(message)s')
    fh = logging.FileHandler(os.path.join(params.output, params.log_id + '.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    # Build Estimator
    for k,v in sorted(params.values().items()):
        tf.logging.info("%s - %s" % (k, v))
    config = run_config(params)
    if params.use_memory and params.restore_memory_file != "":
        warm_start_from = tf.estimator.WarmStartSettings(params.restore_memory_file, vars_to_warm_start='.*memory.*')
    else:
        warm_start_from = None
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=params.output, 
        params=params, config=config, warm_start_from=warm_start_from)

    # Build input
    input_fn_train, train_num = dataset.get_train_eval_input("train", params)
    input_fn_val, val_num = dataset.get_train_eval_input("val", params)

    # train and eval
    train_spec = tf.estimator.TrainSpec(input_fn=input_fn_train, max_steps=params.train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_val, steps=None, throttle_secs=60)
    tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)


if __name__ == "__main__":
    main(parse_args())
