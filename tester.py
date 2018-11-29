#!/usr/bin/env python3
"""MNIST fully-connected tester."""
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.client import timeline
import os
from hfoptimizer import HFOptimizer
from datetime import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# For commandline arguments
parser = ArgumentParser(
    description="Tester for fully-connected architeture on MNIST",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-o",
    "--opt",
    type=str,
    choices=["adam", "sgd", "hfopt"],
    default="hfopt",
    help="choice of the optimizer",
)
parser.add_argument(
    "-l", "--logdir", type=str, default="./logdir", help="path for logging"
)
parser.add_argument(
    "-d",
    "--stop-delta",
    type=float,
    default=1e-3,
    help="loss difference for early stopping",
)
parser.add_argument(
    "-i",
    "--stop-iter",
    type=int,
    default=5,
    help="max number of iterations to watch for early stopping",
)
args = parser.parse_args()
if args.logdir[-1] != "/":
    args.logdir += "/"
args.logdir += datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Get a dataset object
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

# Model definition
x, y_ = iterator.get_next()
y_ = tf.one_hot(y_, 10)
x = tf.identity(x, name="x")
# h = tf.layers.dense(x, 64, activation=tf.nn.relu)
y = tf.layers.dense(x, 10)
pred = tf.nn.softmax(y)

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
)
tf.summary.scalar("loss", loss)

prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32)) * 100
tf.summary.scalar("accuracy", accuracy)

merged = tf.summary.merge_all()

# For not hogging everything
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
gpu_options = tf.GPUOptions(allow_growth=True)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # For profiling
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    if args.opt == "adam":
        train_step = tf.train.AdamOptimizer().minimize(loss)
    elif args.opt == "sgd":
        train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)
    elif args.opt == "hfopt":
        opt = HFOptimizer(
            sess, loss, y, sess_opts=options, sess_run_metadata=run_metadata
        )

    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()
    # writer = tf.summary.FileWriter(args.logdir)

    # Training loop
    sess.run(train_init_op)
    prev_loss = 1e8
    stop_iter = args.stop_iter
    try:
        for step in range(1000):
            if args.opt == "hfopt":
                opt.minimize({})
            else:
                sess.run(
                    train_step, options=options, run_metadata=run_metadata
                )
            summary, curr_loss = sess.run([merged, loss])
            # writer.add_summary(summary, step)
            print("\rStep {} done".format(step + 1), end="")
            if step % 10 == 0:
                saver.save(sess, os.path.join(args.logdir, "model.ckpt"), step)
            if prev_loss - curr_loss < args.stop_delta:
                stop_iter -= 1
                if stop_iter <= 0:
                    break
            else:
                stop_iter = args.stop_iter
                prev_loss = curr_loss
    except KeyboardInterrupt:
        pass
    print("")

    # Test evaluation
    sess.run(test_init_op)
    print("Test Accuracy = {}%".format(sess.run(accuracy)))

    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open(args.logdir + "/timeline_01.json", "w") as f:
        f.write(chrome_trace)
