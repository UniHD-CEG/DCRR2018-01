#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar10-resnet.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import argparse
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.varreplace import remap_variables

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from quantizer import get_quantizer

"""
CIFAR10 ResNet example. See:
Deep Residual Learning for Image Recognition, arxiv:1512.03385
This implementation uses the variants proposed in:
Identity Mappings in Deep Residual Networks, arxiv:1603.05027

I can reproduce the results on 2 TitanX for
n=5, about 7.1% val error after 67k steps (20.4 step/s)
n=18, about 5.95% val error after 80k steps (5.6 step/s, not converged)
n=30: a 182-layer network, about 5.6% val error after 51k steps (3.4 step/s)
This model uses the whole training set instead of a train-val split.

To train:
    ./cifar10-resnet.py --gpu 0,1
"""

# Ternary weights, 8 bit fixed-point activations
BITW = 2
BITA = 8
## Single-precision floating point
# BITW = 32
# BITA = 32
## BNN
# BITW = 1
# BITA = 1

BATCH_SIZE = 128
NUM_UNITS = None


class Model(ModelDesc):

    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 128.0
        assert tf.test.is_gpu_available()
        #image = tf.transpose(image, [0, 3, 1, 2])

	fw, fa = get_quantizer(BITW, BITA)

	def new_get_variable(v):
            name = v.op.name
            # don't quantize first and last layer
            if not name.endswith('W') or 'conv0' in name or 'fct' in name:
                return v
            else:
                logger.info("Quatizing weight {}".format(v.op.name))
                return fw(v)

        def nonlin(x):
            if BITA == 32:
                return tf.nn.relu(x)    # still use relu for 32bit cases
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            return fa(nonlin(x))

        def residual(name, l, increase_dim=False, first=False):
            shape = l.get_shape().as_list()
            in_channel = shape[3]

            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
            else:
                out_channel = in_channel
                stride1 = 1

            with tf.variable_scope(name) as scope:
                if first:
                    b1 = l
                else:
                    b1 = BatchNorm('bn1', l)
                    b1 = activate(b1)

                c1 = Conv2D('conv1', b1, out_channel, stride=stride1)
                b2 = BatchNorm('bn2', c1)
                b2 = activate(b2)
                c2 = Conv2D('conv2', b2, out_channel)
                if increase_dim:
                    l = AvgPooling('pool', l, 2)
                    l = tf.pad(l, [[0,0], [0,0], [0,0], [in_channel//2, in_channel//2]])

                l = c2 + l
                return l

        with remap_variables(new_get_variable), \
                argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling]), \
                argscope(Conv2D, nl=tf.identity, use_bias=False, kernel_shape=3,
                         W_init=variance_scaling_initializer(mode='FAN_OUT')):
            l = Conv2D('conv0', image, 16)
            l = BatchNorm('bn0', l)
            l = activate(l)
            l = residual('res1.0', l, first=True)
            for k in range(1, self.n):
                l = residual('res1.{}'.format(k), l)
            # 32,c=16

            l = residual('res2.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res2.{}'.format(k), l)
            # 16,c=32

            l = residual('res3.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res3.' + str(k), l)
            l = BatchNorm('bnlast', l)
            l = tf.nn.relu(l)
            # 8,c=64
            l = GlobalAvgPooling('gap', l)

        logits = FullyConnected('fct', l, out_dim=10, nl=tf.identity)
        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label)
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.01, summary=True)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('-n', '--num_units',
                        help='number of units in each stage',
                        type=int, default=18)
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    NUM_UNITS = args.num_units

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir()

    dataset_train = get_data('train')
    dataset_test = get_data('test')

    config = TrainConfig(
        model=Model(n=NUM_UNITS),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError()]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (82, 0.01), (123, 0.001), (300, 0.0002)])
        ],
        max_epoch=400,
        nr_tower=max(get_nr_gpu(), 1),
        session_init=SaverRestore(args.load) if args.load else None
    )
    SyncMultiGPUTrainerParameterServer(config).train()
