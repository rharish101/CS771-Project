#!/usr/bin/env python3
from __future__ import print_function
import tensorflow as tf
import os
from hfoptimizer import HFOptimizer
from datetime import datetime

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (mnist.train.images, mnist.train.labels)
    )
    .shuffle(10000)
    .batch(len(mnist.train.labels))
    .repeat()
)
test_dataset = (
    tf.data.Dataset.from_tensor_slices((mnist.test.images, mnist.test.labels))
    .shuffle(10000)
    .batch(len(mnist.test.labels))
)
iterator = tf.data.Iterator.from_structure(
    train_dataset.output_types, train_dataset.output_shapes
)
train_init_op = iterator.make_initializer(train_dataset)
test_init_op = iterator.make_initializer(test_dataset)

LOG_DIR = "logdir/" + datetime.now().strftime("%d-%m-%Y %H:%M:%S")

x, y_ = iterator.get_next()
y_ = tf.one_hot(y_, 10)
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
x = tf.identity(x, name="x")
y = tf.matmul(x, W) + b
pred = tf.nn.softmax(y)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
)
loss = tf.summary.scalar("loss", cross_entropy)
# train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32)) * 100
acc_summ = tf.summary.scalar("accuracy", accuracy)

sess = tf.Session()
opt = HFOptimizer(sess, cross_entropy, y)
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
writer = tf.summary.FileWriter(LOG_DIR)

sess.run(train_init_op)
prev_loss = 1e10
EARLY_STOP = 2e-3
try:
    for step in range(1000):
        # sess.run(train_step)
        opt.minimize({})
        summary, curr_loss, acc = sess.run([loss, cross_entropy, acc_summ])
        writer.add_summary(summary, step)
        print("\rStep {} done".format(step + 1), end="")
        if step % 10 == 0:
            saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), step)
        if prev_loss - curr_loss < EARLY_STOP:
            break
except KeyboardInterrupt:
    pass
print("\n")

sess.run(test_init_op)
print("Test Accuracy = {}%".format(sess.run(accuracy)))
