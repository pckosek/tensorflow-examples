import tensorflow as tf
import os

# -- for the class object --
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
# from tensorflow.python.layers import utils
# from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
# from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
# from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import standard_ops
# from tensorflow.python.util.tf_export import tf_export


# --------------------------------------------------- #
# LOG HELPER FUNCTION
# --------------------------------------------------- #
def shape_log(tensor):
  print("{} has shape: {}".format(
    tensor.op.name,
    tensor.get_shape().as_list()))


# --------------------------------------------------- #
# WEIGHT BIAS EXTRACTION
# --------------------------------------------------- #
def get_layer_vars(op_name) :
  w = tf.get_default_graph().get_tensor_by_name(
      "{}/kernel:0".format(op_name))
  b = tf.get_default_graph().get_tensor_by_name(
      "{}/bias:0".format(op_name))
  return w, b


# --------------------------------------------------- #
# TRAINING STEP LOGGER
# --------------------------------------------------- #
def display_update(indx, interval):
    # post an update
    if ((indx+1)%interval == 0):
        print("[step: {}]".format(indx))

# --------------------------------------------------- #
# SAVER CLASS
# --------------------------------------------------- #
class saver_ops():

  def __init__(self, max_to_keep, global_step, sess):
    self.saver_dir   = os.getcwd()
    self.ckpt_path   = os.path.join(self.saver_dir, 'model.ckpt')
    self.saver_path  = os.path.join(self.ckpt_path, 'saver')
    self.global_step = global_step
    self.sess        = sess

    self.Saver = tf.train.Saver(max_to_keep=max_to_keep)

    self.try_to_restore()

  def try_to_restore(self) :
    # possibly restore model
    ckpt = tf.train.get_checkpoint_state(self.saver_dir)
    if ckpt and ckpt.model_checkpoint_path:
      self.Saver.restore(self.sess, ckpt.model_checkpoint_path)
      print('saver restored')
    else:
      print('saver NOT restored')


  def save(self) :
    # save the saver
    self.Saver.save(self.sess, self.ckpt_path, global_step=self.global_step)


# --------------------------------------------------- #
# DISTRIBUTED DENSE
# --------------------------------------------------- #
class Distributed_Dense(base.Layer):

  def __init__(self, 
               units,
               max_time,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(Distributed_Dense, self).__init__(trainable=trainable, name=name,
                                activity_regularizer=activity_regularizer,
                                **kwargs)
    self.units = units
    self.max_time = max_time
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint
    self.input_spec = base.InputSpec(min_ndim=2)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = base.InputSpec(min_ndim=2,
                                     axes={-1: input_shape[-1].value})
    self.kernel = self.add_variable('kernel',
                                    shape=[1, self.units],
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint,
                                    dtype=self.dtype,
                                    trainable=True)
    self.kernel_ones = tf.ones([input_shape[-1].value, 1], dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_variable('bias',
                                    shape=[1, 1],
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    dtype=self.dtype,
                                    trainable=True)
      self.bias_ones = tf.ones([self.max_time, 1], dtype=self.dtype)
    else:
      self.bias = None
      self.bias_ones = None
    self.built = True

  def call(self, inputs):
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    shape = inputs.get_shape().as_list()
    if len(shape) > 2:      
      # Broadcasting is required for the inputs.
      kernel_op = tf.matmul(self.kernel_ones, self.kernel)
      outputs   = standard_ops.tensordot(inputs, kernel_op, [[len(shape) - 1],
                                                             [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:
      kernel_op = tf.matmul(self.kernel_ones, self.kernel)
      outputs   = tf.matmul(inputs, kernel_op)
    if self.use_bias:
      bias_op = tf.matmul(self.bias_ones, self.bias)
      outputs = tf.add(outputs, bias_op)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if input_shape[-1].value is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)


def distributed_dense(
    inputs, units, max_time,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=init_ops.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None):

  layer = Distributed_Dense(units, max_time,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                trainable=trainable,
                name=name,
                dtype=inputs.dtype.base_dtype,
                _scope=name,
                _reuse=reuse)
  return layer.apply(inputs)