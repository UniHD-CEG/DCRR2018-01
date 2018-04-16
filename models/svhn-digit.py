#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: svhn-digit-dorefa.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import argparse
import numpy as np
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.varreplace import remap_variables
import tensorflow as tf

from quantizer import get_quantizer

"""
This is a tensorpack script for the SVHN results in paper:
DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
http://arxiv.org/abs/1606.06160

To Run:
    ./svhn-digit.py --gpu 0
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

class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 40, 40, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        is_training = get_current_tower_context().is_training

        fw, fa = get_quantizer(BITW, BITA)

        old_get_variable = tf.get_variable

        # monkey-patch tf.get_variable to apply fw
        def quantize_weight(v):
            name = v.op.name
            # don't quantize first and last layer
            if not name.endswith('W') or 'conv0' in name or 'fc' in name:
                return v
            else:
                logger.info("Quantizing weight {}".format(v.op.name))
                return fw(v)

        def nonlin(x):
            if BITA == 32:
                return tf.nn.relu(x)    # still use relu for 32bit cases
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
           return fa(nonlin(x))

        image = image / 256.0

        with remap_variables(quantize_weight), \
                argscope(BatchNorm, decay=0.9, epsilon=1e-4), \
                argscope(Conv2D, use_bias=False, nl=tf.identity):
            logits = (LinearWrap(image)
                      .Conv2D('conv0', 48, 5, padding='VALID', use_bias=True)
                      .MaxPooling('pool0', 2, padding='SAME')
                      .apply(activate)
                      # 18
                      .Conv2D('conv1', 64, 3, padding='SAME')
                      .BatchNorm('bn1').apply(activate)

                      .Conv2D('conv2', 64, 3, padding='SAME')
                      .BatchNorm('bn2')
                      .MaxPooling('pool1', 2, padding='SAME')
                      .apply(activate)
                      # 9
                      .Conv2D('conv3', 128, 3, padding='VALID')
                      .BatchNorm('bn3').apply(activate)
                      # 7

                      .Conv2D('conv4', 128, 3, padding='SAME')
                      .BatchNorm('bn4').apply(activate)

                      .Conv2D('conv5', 128, 3, padding='VALID')
                      .BatchNorm('bn5').apply(activate)
                      # 5
                      .tf.nn.dropout(0.5 if is_training else 1.0)
                      .Conv2D('conv6', 512, 5, padding='VALID')
                      .BatchNorm('bn6')
                      .apply(nonlin)
                      .FullyConnected('fc1', 10, nl=tf.identity)())
        prob = tf.nn.softmax(logits, name='output')

        # compute the number of failed samples
        wrong = prediction_incorrect(logits, label, 1, name='wrong')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(1e-7))

        add_param_summary(('.*/W', ['histogram', 'rms']))
        self.cost = tf.add_n([cost, wd_cost], name='cost')
        add_moving_summary(cost, wd_cost, self.cost)

    def _get_optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=4721 * 100,
            decay_rate=0.5, staircase=True, name='learning_rate')
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr, epsilon=1e-5)


def get_config():
    logger.auto_set_dir()

    # prepare dataset
    d1 = dataset.SVHNDigit('train')
    d2 = dataset.SVHNDigit('extra')
    data_train = RandomMixData([d1, d2])
    data_test = dataset.SVHNDigit('test')

    augmentors = [
        imgaug.Resize((40, 40)),
        imgaug.Brightness(30),
        imgaug.Contrast((0.5, 1.5)),
        # imgaug.GaussianDeform(  # this is slow but helpful. only use it when you have lots of cpus
        # [(0.2, 0.2), (0.2, 0.8), (0.8,0.8), (0.8,0.2)],
        # (40,40), 0.2, 3),
    ]
    data_train = AugmentImageComponent(data_train, augmentors)
    data_train = BatchData(data_train, 128)
    data_train = PrefetchDataZMQ(data_train, 5)

    augmentors = [imgaug.Resize((40, 40))]
    data_test = AugmentImageComponent(data_test, augmentors)
    data_test = BatchData(data_test, 128, remainder=True)

    return TrainConfig(
        dataflow=data_train,
        callbacks=[
            ModelSaver(),
            MinSaver('val-error'),
            InferenceRunner(data_test,
                            [ScalarStats('cost'),
                             ClassificationError('wrong', 'val-error')])
        ],
        model=Model(),
        max_epoch=200,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the GPU to use')
    parser.add_argument('--load', help='load a checkpoint')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    config.nr_tower = max(get_nr_gpu(), 1)
    QueueInputTrainer(config).train()
