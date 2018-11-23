# CS771-Project

This repository hold an implementation of the Hessian-Free optimizer by Martens (2010) in Tensorflow 1.12.
This optimizer is derived from the `tf.train.Optimizer` class, and thus it can be attached to any computation graph which uses only dense variables.

**NOTE**: This has not been tested on Eager execution, and most probably won't work at all.
