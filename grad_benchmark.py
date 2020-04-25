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
from tensorflow.python.ops import rnn
import timeit

from dnc import access
from dnc import addressing
from dnc import dnc
from dnc import util

SEC_TO_MICRO = 1000000
SEC_TO_MILLI = 1000

X, Y, controut, N, W, R, B = 6, 5, 64, 16, 64, 4, 16
d = {
     "kr": [B, R, W],
     "betar": [B, R],
     "kw": [B, 1, W],
     "betaw": [B, 1],
     "wr": [B, R, N],
     "ww": [B, 1, N],
     "u": [B, N],
     "f": [B, R],
     "g": [B, 1],
     "readmode": [B, R, 3],
     "link": [B, N, N],
     "mem": [B, N, W],
     "erase": [B, 1, W],
     "xi": [1, B, controut],
     }


def bench_method(method, inputdims, reps=1000, **kwargs):
    inputs = [tf.placeholder(tf.float32, i) for i in inputdims]
    output = method(*inputs, **kwargs)
    grad = tf.gradients(output, inputs)
    data = [np.random.randn(*i) for i in inputdims]
    with tf.Session() as sess:
        forward_time = timeit.timeit(lambda: sess.run(
            output,
            feed_dict={i: d for (i, d) in zip(inputs, data)}), number=reps)
        print("forward: " + str(SEC_TO_MICRO*forward_time/reps) + " micros")
    with tf.Session() as sess:
        gradient_time = timeit.timeit(lambda: sess.run(
            grad,
            feed_dict={i: d for (i, d) in zip(inputs, data)}), number=reps)
        print("gradient: " + str(SEC_TO_MICRO*gradient_time/reps) + " micros")


def bench_usage(module, reps=1000):
    ww = tf.placeholder(tf.float32, d["ww"])
    f = tf.placeholder(tf.float32, d["f"])
    wr = tf.placeholder(tf.float32, d["wr"])
    u = tf.placeholder(tf.float32, d["u"])
    output = module(ww, f, wr, u)
    grad = tf.gradients(output, [f, wr, u])
    ww_data = np.random.randn(*d["ww"])
    f_data = np.random.randn(*d["f"])
    wr_data = np.random.randn(*d["wr"])
    u_data = np.random.randn(*d["u"])
    with tf.Session() as sess:
        forward_time = timeit.timeit(lambda: sess.run(
            output,
            feed_dict={
                ww: ww_data,
                f: f_data,
                wr: wr_data,
                u: u_data}), number=reps)
        print("forward: " + str(SEC_TO_MICRO*forward_time/reps) + " micros")
    with tf.Session() as sess:
        gradient_time = timeit.timeit(lambda: sess.run(
            grad,
            feed_dict={
                ww: ww_data,
                f: f_data,
                wr: wr_data,
                u: u_data}), number=reps)
        print("gradient: " + str(SEC_TO_MICRO*gradient_time/reps) + " micros")



def bench_access(ma, method, inputdims, reps=1000):
    inputs = [tf.placeholder(tf.float32, i) for i in inputdims]
    contout = ma._read_inputs(inputs[0])
    output = method(contout, *inputs[1:])
    grad = tf.gradients(output, inputs)
    data = [np.random.randn(*i) for i in inputdims]
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        forward_time = timeit.timeit(lambda: sess.run(
            output,
            feed_dict={i: d for (i, d) in zip(inputs, data)}), number=reps)
        print("forward: " + str(SEC_TO_MICRO*forward_time/reps) + " micros")
    with tf.Session() as sess:
        sess.run(init)
        gradient_time = timeit.timeit(lambda: sess.run(
            grad,
            feed_dict={i: d for (i, d) in zip(inputs, data)}), number=reps)
        print("gradient: " + str(SEC_TO_MICRO*gradient_time/reps) + " micros")



def bench_memory_access(reps=1000):
    ma = access.MemoryAccess(N, W, R, 1)
    initial_state = ma.initial_state(B)
    xi = tf.placeholder(tf.float32, d["xi"])
    xi_data = np.random.randn(*d["xi"])
    output, _ = rnn.dynamic_rnn(
                    cell=ma,
                    inputs=xi,
                    initial_state=initial_state,
                    time_major=True)
    init = tf.global_variables_initializer()
    grad = tf.gradients(output, [xi])
    with tf.Session() as sess:
        sess.run(init)
        forward_time = timeit.timeit(lambda: sess.run(
            output,
            feed_dict={xi: xi_data}), number=reps)
        print("forward: " + str(SEC_TO_MILLI*forward_time/reps) + " ms")
    with tf.Session() as sess:
        sess.run(init)
        gradient_time = timeit.timeit(lambda: sess.run(
            grad,
            feed_dict={xi: xi_data}), number=reps)
        print("gradient: " + str(SEC_TO_MILLI*gradient_time/reps) + " ms")

def bench_computer(reps=1000):
    access_config = {
        "memory_size": N,
        "word_size": W,
        "num_reads": R,
        "num_writes": 1,
    }
    controller_config = {
        "hidden_size": controut,
    }
    clip_value = 20
    output_size = Y
    input_sequence = tf.placeholder(tf.float32, [1, B, X])
    input_data = np.random.randn(1, B, X)
    dnc_core = dnc.DNC(access_config, controller_config, output_size, clip_value)
    initial_state = dnc_core.initial_state(B)
    output, _ = rnn.dynamic_rnn(
          cell=dnc_core,
          initial_state=initial_state,
          inputs=input_sequence,
          time_major=True)
    trainable_variables = tf.trainable_variables()
    grad =  tf.gradients(output, input_sequence)#trainable_variables)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        forward_time = timeit.timeit(lambda: sess.run(
            output,
            feed_dict={input_sequence: input_data}), number=reps)
        print("forward: " + str(SEC_TO_MILLI*forward_time/reps) + " ms")
    with tf.Session() as sess:
        sess.run(init)
        gradient_time = timeit.timeit(lambda: sess.run(
            grad,
            feed_dict={input_sequence: input_data}), number=reps)
        print("gradient: " + str(SEC_TO_MILLI*gradient_time/reps) + " ms")




def main():
    print("### ADDRESSING ###")
    print("contentaddress")
    contaddr = addressing.CosineWeights(R, W)
    bench_method(contaddr, [d["mem"], d["kr"], d["betar"]])
    freeness = addressing.Freeness(N)
    print("memoryretention")
    bench_method(freeness._usage_after_write, [d["u"], d["ww"]])
    print("usage")
    bench_usage(freeness)
    templink = addressing.TemporalLinkage(N, 1)
    print("precedenceweight")
    bench_method(templink._precedence_weights, [d["ww"], d["ww"]])
    print("forwardweight")
    bench_method(templink.directional_read_weights, [d["link"], d["wr"]], forward=True)
    print("backwardweight")
    bench_method(templink.directional_read_weights, [d["link"], d["wr"]], forward=False)
    print("### ACCESS ###")
    print("eraseandadd")
    bench_method(access._erase_and_write, [d["mem"], d["ww"], d["erase"], d["erase"]])
    print("readmem")
    bench_method(tf.matmul, [d["wr"], d["mem"]])
    ma = access.MemoryAccess(N, W, R, 1)
    print("writeweights")
    bench_access(ma, ma._write_weights, [[B, controut], d["mem"], d["u"]])
    print("readweights")
    bench_access(ma, ma._read_weights, [[B, controut], d["mem"], d["wr"], d["link"]])
    print("MemoryAccess")
    bench_memory_access()
    print("### COMPUTER ###")
    bench_computer()

if __name__ == "__main__":
    main()

