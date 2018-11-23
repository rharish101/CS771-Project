#!/usr/bin/env python3
from __future__ import print_function
import tensorflow as tf
from hfoptimizer import HFOptimizer

a = tf.get_variable("test", shape=[], initializer=tf.zeros_initializer)
loss = (5 - a) ** 2
# opt = tf.train.AdamOptimizer()
# train_step = opt.minimize(loss)
# opt = HFOptimizer(batch_size=1, cg_max_iters=10, learning_rate=1e-1)
# train_step = opt.minimize(loss, a, verbose=True)

config = tf.ConfigProto(device_count={"GPU": 0})
sess = tf.Session(config=config)
opt = HFOptimizer(sess, loss, a)
sess.run(tf.global_variables_initializer())

# writer = tf.summary.FileWriter(
# logdir="logdir/" + datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
# graph=sess.graph,
# )

for i in range(10):
    # sess.run(train_step)
    opt.minimize({})
    print("\rEpoch: {}, value: {}".format(i + 1, sess.run(a)), end="")
print("")
