#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 09:50:10 2020

@author: sondre
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sonnet as snt
import tensorflow as tf
import timeit

from dnc import addressing
from dnc import util


X, Y, controut, N, W, R, B = 6, 5, 64, 16, 64, 4, 16
d = {
     "kr": [B, R, W],
     "betar": [B, R],
     "kw": [B, 1, W],
     "betaw": [B, 1],
     "wr": [B, R, N],
     "ww": [B, 1, N],
     "u": [B, N],
     "f": [B, 1],
     "g": [B, 1],
     "readmode": [B, R, 3],
     "link": [B, N, N],
     "mem": [B, N, W],
     "erase": [B, W],
     }


def bench_method(method, inputdims, reps=1000):
    inputs = [tf.placeholder(tf.float32, i) for i in inputdims]
    output = method(*inputs)
    grad = tf.gradients(output, inputs)
    data = [np.random.randn(*i) for i in inputdims]
    with tf.Session() as sess:
        forward_time = timeit.timeit(lambda: sess.run(
            output,
            feed_dict={i: d for (i, d) in zip(inputs, data)}), number=reps)
        gradient_time = timeit.timeit(lambda: sess.run(
            grad,
            feed_dict={i: d for (i, d) in zip(inputs, data)}), number=reps)
        print("forward: " + str(1000000*forward_time/reps) + " micros")
        print("gradient: " + str(1000000*gradient_time/reps) + " micros")


def bench_cosineweights():
    module = addressing.CosineWeights(R, W)
    mem = tf.placeholder(tf.float32, d["mem"])
    key = tf.placeholder(tf.float32, d["kr"])
    beta = tf.placeholder(tf.float32, d["betar"])
    output = module(mem, key, beta)
    grad = tf.gradients(output, [mem, key, beta])

    mem_data = np.random.randn(*d["mem"])
    key_data = np.random.randn(*d["kr"])
    beta_data = np.random.randn(*d["betar"])

    #logdir = "log/bench"
    #writer = tf.summary.FileWriter(logdir, grad)

    with tf.Session() as sess:
   #     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #    run_metadata = tf.RunMetadata()
        reps = 10000
        total_time = timeit.timeit(lambda: sess.run(
                 output,
    #            options=run_options,
    #            run_metadata=run_metadata,
                feed_dict={mem: mem_data,
                           key: key_data,
                           beta: beta_data}), number=reps)
        print("Total time: " + str(total_time) + "s")
        print("Per iter: " + str(total_time*1000000/reps) + "micros")
     #   writer.add_run_metadata(run_metadata)
     #   print(run_metadata)

def main():
    module = addressing.CosineWeights(R, W)
    print("contentaddress")
    bench_method(module, [d["mem"], d["kr"], d["betar"]])

if __name__ == "__main__":
    main()


