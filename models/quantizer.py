#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dorefa.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from tensorpack.utils.argtools import graph_memoized


@graph_memoized
def get_quantizer(bitW, bitA):
    G = tf.get_default_graph()

    def quantize(x, k):
        n = float(2**k - 1)
        with G.gradient_override_map({"Round": "Identity"}):
            return tf.round(x * n) / n

    def tw_ternarize(x, thre):
        shape = x.get_shape()

        thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thre)

        w_p = tf.get_variable('Wp', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'positives'], initializer=1.0)
        w_n = tf.get_variable('Wn', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'negatives'], initializer=1.0)

        mask = tf.ones(shape)
        mask_p = tf.where(x > thre_x, tf.ones(shape) * w_p, mask)
        mask_np = tf.where(x < -thre_x, tf.ones(shape) * w_n, mask_p)
        mask_z = tf.where((x < thre_x) & (x > - thre_x), tf.zeros(shape), mask)

        with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
            w =  tf.sign(x) * tf.stop_gradient(mask_z)

        w = w * mask_np

        return w

    def fw(x):
        if bitW == 32:
            return x
        if bitW == 1:   # BWN
            with G.gradient_override_map({"Sign": "Identity"}):
                E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
                return tf.sign(x / E) * E
        if bitW == 2:
            return tw_ternarize(x, 0.05)
        x = tf.tanh(x)
        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
        return 2 * quantize(x, bitW) - 1

    def fa(x):
        if bitA == 32:
            return x
        return quantize(x, bitA)

    return fw, fa
