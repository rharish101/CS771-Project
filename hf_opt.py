"""
Hessian Free Optimizer.

Original Author: MoonLight, 2018
Modified by: rharish, 2018
"""

import tensorflow as tf

try:
    import colored_traceback.auto
except ImportError:
    pass


class clr:
    """Used for color debug output to console."""

    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


class HFOptimizer(tf.train.Optimizer):
    """Tensorflow based Hessian-Free (Truncated Newton) optimizer.

    More details: (Martens, ICML 2010) and (Martens & Sutskever, ICML 2011).

    Methods to use:
    __init__:
        Creates Tensorflow graph and variables.
    minimize:
        Perfoms HF optimization.
    """

    DUMPING_NUMERICAL_ERROR_STOP_FLOAT32 = 1e-4
    CG_NUMERICAL_ERROR_STOP_FLOAT32 = 1e-20
    DUMPING_NUMERICAL_ERROR_STOP_FLOAT64 = 1e-8
    CG_NUMERICAL_ERROR_STOP_FLOAT64 = 1e-80

    def __init__(
        self,
        learning_rate=1,
        cg_decay=0.95,
        damping=0.5,
        adjust_damping=True,
        batch_size=None,
        use_gauss_newton_matrix=True,
        preconditioner=False,
        prec_loss=None,
        gap=10,
        cg_max_iters=50,
        dtype=tf.float32,
    ):
        """Create Tensorflow graph and variables.

        learning_rate: float number
            Learning rate parameter for training neural network.
        cg_decay: float number
            Decay for previous result of computing delta with conjugate
            gradient method for the initialization of next iteration
            conjugate gradient.
        damping: float number
            Initial value of the Tikhonov damping coefficient.
        adjust_damping: bool
            Whether adjust damping parameter dynamically using
            Levenberg-Marquardt heuristic or not.
        batch_size: int number or None
            Used for Jacobian vector product (Rop) computation, necessary if
            used dynamic input batch size.
        use_gauss_newton_matrix: bool
            Whether use Gauss Newton Matrix (True) or Hessian matrix (False) in
            conjugate gradient computation.
        preconditioner: bool
            Martens preconditioner uses The Empirical Fisher Diagonal for its
            computation, don't use it with dynamically adjusting damping
            parameter, because it can cause numerical errors.
            Can be used only with Gauss Newton Matrix.
        prec_loss: Tensorflow tensor object
            Used for computing preconditioner; if using preconditioner it's
            better to set it explicitly. For this parameter use loss before
            reduce sum function over the batch inputs is applied.
        gap: int
            Size of window gap for which delta loss difference is computed,
            used for early stoping in conjugate gradient computation.
        cg_max_iters: int
            Number of maximum iterations of conjugate gradient computations.
        dtype: Tensorflow type
            Type of Tensorflow variables.
        """
        super().__init__(True, "HFOptimizer")

        self.cg_decay = cg_decay
        self.prec_loss = prec_loss
        self.batch_size = batch_size
        self.use_prec = preconditioner
        self.learning_rate = learning_rate
        self.use_gnm = use_gauss_newton_matrix
        self.damping = damping
        self.gap = gap
        self.cg_max_iters = cg_max_iters
        self.adjust_damping = adjust_damping
        self.damp_pl = tf.constant(0.0)
        self.dtype = dtype

        self.cg_num_err = HFOptimizer.CG_NUMERICAL_ERROR_STOP_FLOAT32
        self.damp_num_err = HFOptimizer.DUMPING_NUMERICAL_ERROR_STOP_FLOAT32
        if dtype == tf.float64:
            self.cg_num_err = HFOptimizer.CG_NUMERICAL_ERROR_STOP_FLOAT64
            self.damp_num_err = (
                HFOptimizer.DUMPING_NUMERICAL_ERROR_STOP_FLOAT64
            )
        if not self.use_gnm:
            self.damp_num_err = 1e-1

        if not self.use_gnm and self.use_prec:
            self.use_prec = False
            print(
                clr.WARNING
                + "WARNING: You set preconditioner to True but "
                + "use_gauss_newton_matrix to False, "
                + "and it's prohibited, so we set preconditioner back to "
                + "False, if you ask why see more information "
                + "on (Martens & Sutskever, ICML 2011)."
                + clr.ENDC
            )
        elif self.use_prec and self.use_gnm and self.prec_loss is None:
            print(
                clr.WARNING
                + "WARNING: If you use preconditioner it is "
                + "better to set prec_loss explicitly, because it can "
                + "cause graph making problem. (What's prec_loss see "
                + "in description)"
                + clr.ENDC
            )

    def info(self):
        """Print initial settings of HF optimizer."""
        print(
            clr.BOLD
            + clr.OKGREEN
            + "Hessian-Free Optimizer initial settings:"
            + clr.ENDC
        )
        print("    CG delta decay: {}".format(self.cg_decay))
        print("    Learning Rate: {}".format(self.learning_rate))
        print("    Initial Tikhonov damping: {}".format(self.damping))
        if self.adjust_damping:
            print(
                "    Optimizer adjusts damping dynamically using "
                + "Levenberg-Marquardt heuristic."
            )
        else:
            print("    Tikhonov damping is static.")
        if self.use_gnm:
            print("    Optimizer uses Gauss-Newton matrix for cg computation.")
        else:
            print("    Optimizer uses Hessian matrix for cg computation.")
        if self.use_prec:
            print("    Optimizer uses preconditioner.")
        print("    Gap of delta loss tracking: {}".format(self.gap))
        print("    Max cg iterations: {}".format(self.cg_max_iters))
        print(clr.OKGREEN + "Optimizer is ready for using." + clr.ENDC)

    def compute_gradients(
        self,
        loss,
        output,
        var_list=None,
        gate_gradients=tf.train.Optimizer.GATE_OP,
        aggregation_method=None,
        colocate_gradients_with_ops=False,
        grad_loss=None,
    ):
        """Compute gradients of `loss` for the variables in `var_list`.

        This is the first part of `minimize()`.  It returns a list
        of (gradient, variable) pairs where "gradient" is the gradient
        for "variable".  Note that "gradient" can be a `Tensor`, an
        `IndexedSlices`, or `None` if there is no gradient for the
        given variable.

        Args:
            loss: A Tensor containing the value to minimize or a callable
                taking no arguments which returns the value to minimize.
            output: Tensorflow tensor object
                Variable with respect to which the Hessian of the objective is
                positive-definite, implicitly defining the Gauss-Newton matrix.
                Typically, it is the activation of the output layer.
            var_list: Optional list or tuple of `tf.Variable` to update to
                minimize `loss`.  Defaults to the list of variables collected
                in the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
            gate_gradients: How to gate the computation of gradients.  Can be
                `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
            aggregation_method: Specifies the method used to combine gradient
                terms.  Valid values are defined in the class
                `AggregationMethod`.
            colocate_gradients_with_ops: If True, try colocating gradients with
                the corresponding op.
            grad_loss: Optional. A `Tensor` holding the gradient computed for
                `loss`.

        Returns:
            A list of (gradient, variable) pairs. Variable is always present,
            but gradient can be `None`.

        Raises:
            TypeError: If `var_list` contains anything else than `Variable`
                objects.
            ValueError: If some arguments are invalid.

        @compatibility(eager)
        Eager execution not supported.
        @end_compatibility

        """
        self.loss = loss
        self.output = output

        # Network weights
        if var_list is None:
            self.W = tf.trainable_variables()
        else:
            self.W = var_list

        grad_and_vars = super().compute_gradients(
            loss=loss,
            var_list=var_list,
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss,
        )

        self.grads = [grad for grad, var in grad_and_vars]
        return grad_and_vars

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(
            initial_value=0, name="cg_step", colocate_with=first_var
        )

        for w in var_list:
            self._zeros_slot(w, "delta", self._name)
            self._zeros_slot(w, "direction", self._name)
            self._zeros_slot(w, "residual", self._name)

    def _prepare(self):
        cg_op, res_norm, dl = self.__conjugate_gradient(self.grads)
        self.ops = {
            "cg_update": cg_op,
            "res_norm": res_norm,
            "dl": dl,
            "set_delta_0": self.__update_delta_0(),
            "train": self.__train_op(),
        }

    def minimize(
        self,
        loss,
        output,
        global_step=None,
        var_list=None,
        gate_gradients=tf.train.Optimizer.GATE_OP,
        aggregation_method=None,
        colocate_gradients_with_ops=False,
        name=None,
        grad_loss=None,
        verbose=False,
    ):
        """Add operations to minimize `loss` by updating `var_list`.

        This method simply combines calls `compute_gradients()` and
        `apply_gradients()`. If you want to process the gradient before
        applying them call `compute_gradients()` and `apply_gradients()`
        explicitly instead of using this function.

        Args:
            loss: A `Tensor` containing the value to minimize.
            output: Tensorflow tensor object
                Variable with respect to which the Hessian of the objective is
                positive-definite, implicitly defining the Gauss-Newton matrix.
                Typically, it is the activation of the output layer.
            global_step: Optional `Variable` to increment by one after the
                variables have been updated.
            var_list: Optional list or tuple of `Variable` objects to update to
                minimize `loss`.  Defaults to the list of variables collected
                in the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
            gate_gradients: How to gate the computation of gradients.  Can be
                `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
            aggregation_method: Specifies the method used to combine gradient
                terms.  Valid values are defined in the class
                `AggregationMethod`.
            colocate_gradients_with_ops: If True, try colocating gradients with
                the corresponding op.
            name: Optional name for the returned operation.
            grad_loss: Optional. A `Tensor` holding the gradient computed for
                `loss`.
            verbose: bool
                If True prints CG iteration number.

        Returns:
            An Operation that updates the variables in `var_list`.  If
            `global_step` was not `None`, that operation also increments
            `global_step`.

        Raises:
            ValueError: If some of the variables are not `Variable` objects.

        @compatibility(eager)
        Eager execution not supported.
        @end_compatibility

        """
        self.verbose = verbose

        grads_and_vars = self.compute_gradients(
            loss,
            output,
            var_list=var_list,
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss,
        )

        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for "
                "ops that do not support gradients, between variables %s and "
                "loss %s." % ([str(v) for _, v in grads_and_vars], loss)
            )

        return self.apply_gradients(
            grads_and_vars, global_step=global_step, name=name
        )

    def _apply_dense(self, grad, var):
        """Perform main training operations.

        grad: Tensorflow tensor object
            List of gradients of loss w.r.t. var
        var: Tensorflow tensor object
            List of variables to train
        """
        self.damp_pl = self.damping
        if self.adjust_damping:
            loss_before_cg = tf.identity(self.loss)

        dl_track = tf.expand_dims(self.ops["dl"], axis=0)

        combined_op_1 = tf.group(
            [loss_before_cg, dl_track[0], self.ops["set_delta_0"]]
        )
        with tf.control_dependencies([combined_op_1]):
            with tf.variable_scope("for_loop"):
                i = tf.constant(0)
                stop = tf.constant(False)

            def loop(i, stop, dl_track):
                if self.verbose:
                    tf.logging.info(
                        clr.OKGREEN
                        + "\r[CG iteration: {}]".format(i)
                        + clr.ENDC
                    )

                k = tf.maximum(self.gap, i // self.gap)

                rn = tf.identity(self.ops["res_norm"])
                with tf.control_dependencies([rn]):
                    stop = tf.cond(
                        rn < self.cg_num_err, lambda: True, lambda: stop
                    )

                cg_update = self.ops["cg_update"]
                with tf.control_dependencies([cg_update]):
                    dl_track = tf.concat(
                        [dl_track, tf.expand_dims(self.ops["dl"], axis=0)],
                        axis=0,
                    )

                def early_stop():
                    margin = (
                        dl_track[i + 1] - dl_track[i + 1 - k]
                    ) / dl_track[i + 1]
                    return tf.cond(
                        tf.logical_and(
                            tf.debugging.is_nan(margin), margin < 1e-4
                        ),
                        lambda: True,
                        lambda: stop,
                    )

                stop = tf.cond(i > k, early_stop, lambda: stop)
                i += 1
                return i, stop, dl_track

            i, stop, dl_track = tf.while_loop(
                lambda i, stop, dl_track: tf.logical_and(
                    i < self.cg_max_iters, stop
                ),
                loop,
                (i, stop, dl_track),
                shape_invariants=(
                    i.get_shape(),
                    stop.get_shape(),
                    tf.TensorShape([None]),
                ),
                parallel_iterations=1,
                maximum_iterations=self.cg_max_iters,
            )
            loop_vars = tf.group([i, stop])

        if self.verbose:
            tf.logging.info("\n")

        if self.adjust_damping:
            self.damp_pl = tf.constant(0.0)
            dl = tf.identity(self.ops["dl"])
            self.damp_pl = self.damping

        combined_op_2 = tf.group([loop_vars, dl, self.ops["train"]])
        with tf.control_dependencies([combined_op_2]):
            if self.adjust_damping:
                loss_after_cg = tf.identity(self.loss)
                reduction_ratio = (loss_after_cg - loss_before_cg) / dl

                def elseif():
                    return tf.cond(
                        tf.logical_and(
                            reduction_ratio > 0.75,
                            self.damping > self.damp_num_err,
                        ),
                        lambda: self.damping / 1.5,
                        lambda: self.damping,
                    )

                self.damping = tf.cond(
                    tf.logical_and(
                        reduction_ratio < 0.25,
                        self.damping > self.damp_num_err,
                    ),
                    lambda: self.damping * 1.5,
                    elseif,
                )

            return tf.group([combined_op_2, self.damping])

    def __conjugate_gradient(self, gradients):
        """Perform conjugate gradient method.

        It minimizes the quadratic equation and find best delta of
        network parameters.

        gradients: list of Tensorflow tensor objects
            Network gradients.

        return: Tensorflow tensor object
            Update operation for delta.
        return: Tensorflow tensor object
            Residual norm, used to prevent numerical errors.
        return: Tensorflow tensor object
            Delta loss.
        """
        with tf.name_scope("conjugate_gradient"):
            cg_update_ops = []

            prec = None
            if self.use_prec:
                if self.prec_loss is None:
                    graph = tf.get_default_graph()
                    lop = self.loss.op.node_def
                    self.prec_loss = graph.get_tensor_by_name(
                        lop.input[0] + ":0"
                    )
                batch_size = None
                if self.batch_size is None:
                    self.prec_loss = tf.unstack(self.prec_loss)
                    batch_size = self.prec_loss.get_shape()[0]
                else:
                    self.prec_loss = [
                        tf.gather(self.prec_loss, i)
                        for i in range(self.batch_size)
                    ]
                    batch_size = len(self.prec_loss)
                prec = [
                    [
                        g ** 2
                        for g in tf.gradients(
                            tf.gather(self.prec_loss, i), self.W
                        )
                    ]
                    for i in range(batch_size)
                ]
                prec = [
                    (sum(tensor) + self.damping) ** (-0.75)
                    for tensor in tf.transpose(tf.constant(prec))
                ]

            Ax = None
            if self.use_gnm:
                Ax = self.__Gv([self.get_slot(w, "delta") for w in self.W])
            else:
                Ax = self.__Hv(
                    gradients, [self.get_slot(w, "delta") for w in self.W]
                )

            b = [-grad for grad in gradients]
            bAx = [b - Ax for b, Ax in zip(b, Ax)]

            condition = tf.equal(
                self._get_non_slot_variable("cg_step", self.W[0].graph), 0
            )
            r = [
                tf.cond(condition, lambda: tf.assign(r, bax), lambda: r)
                for r, bax in zip(
                    [self.get_slot(w, "residual") for w in self.W], bAx
                )
            ]

            d = None
            if self.use_prec:
                d = [
                    tf.cond(condition, lambda: tf.assign(d, p * r), lambda: d)
                    for p, d, r in zip(
                        prec,
                        [self.get_slot(w, "direction") for w in self.W],
                        r,
                    )
                ]
            else:
                d = [
                    tf.cond(condition, lambda: tf.assign(d, r), lambda: d)
                    for d, r in zip(
                        [self.get_slot(w, "direction") for w in self.W], r
                    )
                ]

            Ad = None
            if self.use_gnm:
                Ad = self.__Gv(d)
            else:
                Ad = self.__Hv(gradients, d)

            residual_norm = tf.reduce_sum([tf.reduce_sum(r ** 2) for r in r])

            alpha = tf.reduce_sum(
                [tf.reduce_sum(d * ad) for d, ad in zip(d, Ad)]
            )
            alpha = residual_norm / alpha

            if self.use_prec:
                beta = tf.reduce_sum(
                    [
                        tf.reduce_sum(p * (r - alpha * ad) ** 2)
                        for r, ad, p in zip(r, Ad, prec)
                    ]
                )
            else:
                beta = tf.reduce_sum(
                    [
                        tf.reduce_sum((r - alpha * ad) ** 2)
                        for r, ad in zip(r, Ad)
                    ]
                )

            self.beta = beta
            beta = beta / residual_norm

            for i, w in reversed(list(enumerate(self.W))):
                delta = self.get_slot(w, "delta")
                update_delta = tf.assign(
                    delta, delta + alpha * d[i], name="update_delta"
                )
                update_residual = tf.assign(
                    self.get_slot(self.W[i], "residual"),
                    r[i] - alpha * Ad[i],
                    name="update_residual",
                )
                p = 1.0
                if self.use_prec:
                    p = prec[i]
                update_direction = tf.assign(
                    self.get_slot(self.W[i], "direction"),
                    p * (r[i] - alpha * Ad[i]) + beta * d[i],
                    name="update_direction",
                )
                cg_update_ops.append(update_delta)
                cg_update_ops.append(update_residual)
                cg_update_ops.append(update_direction)

            with tf.control_dependencies(cg_update_ops):
                cg_update_ops.append(
                    tf.assign_add(
                        self._get_non_slot_variable(
                            "cg_step", self.W[0].graph
                        ),
                        1,
                    )
                )
            cg_op = tf.group(*cg_update_ops)

        dl = tf.reduce_sum(
            [
                tf.reduce_sum(
                    0.5 * (delta * ax) + grad * self.get_slot(w, "delta")
                )
                for w, grad, ax in zip(self.W, gradients, Ax)
            ]
        )

        return cg_op, residual_norm, dl

    def __Hv(self, grads, vec):
        """Compute Hessian vector product.

        grads: list of Tensorflow tensor objects
            Network gradients.
        vec: list of Tensorflow tensor objects
            Vector that is multiplied by the Hessian.

        return: list of Tensorflow tensor objects
            Result of multiplying Hessian by vec.
        """
        grad_v = [tf.reduce_sum(g * v) for g, v in zip(grads, vec)]
        Hv = tf.gradients(grad_v, self.W, stop_gradients=vec)
        Hv = [hv + self.damp_pl * v for hv, v in zip(Hv, vec)]

        return Hv

    def __Gv(self, vec):
        """Compute the product G by vec = JHJv (G is the Gauss-Newton matrix).

        vec: list of Tensorflow tensor objects
            Vector that is multiplied by the Gauss-Newton matrix.

        return: list of Tensorflow tensor objects
            Result of multiplying Gauss-Newton matrix by vec.
        """
        Jv = self.__Rop(self.output, self.W, vec)
        Jv = tf.reshape(tf.stack(Jv), [-1, 1])
        H = tf.transpose(tf.gradients(self.loss, self.output)[0])
        if len(H.get_shape().as_list()) < 2:
            HJv = tf.gradients(H * Jv, self.output, stop_gradients=Jv)[0]
            JHJv = tf.gradients(
                tf.transpose(HJv) * self.output, self.W, stop_gradients=HJv
            )
        else:
            HJv = tf.gradients(
                tf.matmul(H, Jv), self.output, stop_gradients=Jv
            )[0]
            JHJv = tf.gradients(
                tf.matmul(tf.transpose(HJv), self.output),
                self.W,
                stop_gradients=HJv,
            )
        JHJv = [gv + self.damp_pl * v for gv, v in zip(JHJv, vec)]

        return JHJv

    def __Rop(self, f, x, vec):
        """Compute Jacobian vector product.

        f: Tensorflow tensor object
            Objective function.
        x: list of Tensorflow tensor objects
            Parameters with respect to which computes Jacobian matrix.
        vec: list of Tensorflow tensor objects
            Vector that is multiplied by the Jacobian.

        return: list of Tensorflow tensor objects
            Result of multiplying Jacobian (df/dx) by vec.
        """
        r = None
        if self.batch_size is None:
            try:
                r = [
                    tf.reduce_sum(
                        [
                            tf.reduce_sum(v * tf.gradients(f, x)[i])
                            for i, v in enumerate(vec)
                        ]
                    )
                    for f in tf.unstack(f)
                ]
            except ValueError:
                assert False, (
                    clr.FAIL + clr.BOLD + "Batch size is None, but used "
                    "dynamic shape for network input, set proper "
                    "batch_size in HFOptimizer initialization" + clr.ENDC
                )
        else:
            if len(f.get_shape().as_list()) == 0:
                fn = tf.reshape(f, [1])
            else:
                fn = f
            r = [
                tf.reduce_sum(
                    [
                        tf.reduce_sum(v * tf.gradients(tf.gather(fn, i), x)[j])
                        for j, v in enumerate(vec)
                    ]
                )
                for i in range(self.batch_size)
            ]

        assert r is not None, (
            clr.FAIL
            + clr.BOLD
            + "Something went wrong in Rop computation"
            + clr.ENDC
        )

        return r

    def __update_delta_0(self):
        """Update initial delta for conjugate gradient method.

        The old delta is multiplied by cg_decay.

        return: list of Tensorflow tensor objects
            Update initial delta operation.
        """
        update_delta_0_ops = []
        for w in self.W:
            delta = self.get_slot(w, "delta")
            update_delta = tf.assign(delta, self.cg_decay * delta)
            update_delta_0_ops.append(update_delta)
        update_delta_0_op = tf.group(*update_delta_0_ops)

        return update_delta_0_op

    def __train_op(self):
        """Perform main training operation, i.e. updates weights.

        return: list of Tensorflow tensor objects
            Main training operations
        """
        update_ops = []
        for w in reversed(self.W):
            with tf.control_dependencies(update_ops):
                update_ops.append(
                    tf.assign(
                        w, w + self.learning_rate * self.get_slot(w, "delta")
                    )
                )
        training_op = tf.group(*update_ops)

        return training_op
