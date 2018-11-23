#!/usr/bin/env python3
from __future__ import print_function
import tensorflow as tf
from hf_opt import HFOptimizer
from datetime import datetime

a = tf.get_variable("test", shape=[], initializer=tf.zeros_initializer)
loss = (5 - a) ** 2
# opt = tf.train.AdamOptimizer()
opt = HFOptimizer(batch_size=1, cg_max_iters=10)
train_step = opt.minimize(loss, a, verbose=True)

config = tf.ConfigProto(device_count={"GPU": 0})
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# writer = tf.summary.FileWriter(
# logdir="logdir/" + datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
# graph=sess.graph,
# )

for i in range(2):
    sess.run(train_step)
print(sess.run(a))
