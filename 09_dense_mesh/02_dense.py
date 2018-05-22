# --------------------------------------------------- #
# IMPORT STATEMENTS
# --------------------------------------------------- #
import tensorflow as tf
import sys
import os
import numpy as np
import matlab_ops as m

# disable tensorflow logging stuff
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# --------------------------------------------------- #
# WEIGHT BIAS EXTRACTION
# --------------------------------------------------- #
def get_dense_vars(op_name) :
  w = tf.get_default_graph().get_tensor_by_name(
      "{}/kernel:0".format(op_name))
  b = tf.get_default_graph().get_tensor_by_name(
      "{}/bias:0".format(op_name))
  return w, b

# --------------------------------------------------- #
# TRAINING METHOD
# --------------------------------------------------- #

def train():
  c = constants()

  some_example_random_number = -0.31415

  x      = tf.constant(some_example_random_number, dtype=tf.float32)
  x      = tf.reshape(x, [1,1])
  d      = tf.layers.dense(x, units=2, name='dense_1')
  e      = tf.layers.dense(d, units=1)

  weights, biases = get_dense_vars('dense_1') 

  print(d)
  print(weights)
  print(biases)

  # https://stackoverflow.com/questions/45372291/how-to-get-weights-in-tf-layers-dense

  labels = tf.constant(1.0, dtype=tf.float32)
  loss   = tf.reduce_sum(tf.square(e-labels))

  train_op = tf.train.AdamOptimizer(learning_rate=0.5).minimize(loss)
  init_op  = tf.global_variables_initializer()

  # connect to matlab
  eng = m.matlab_connection(c.MATLAB_SESSION)

  with tf.Session() as sess:

      # initialize global variables
      sess.run(init_op)

      xs = []
      ws = []
      bs = []

      # run the training operation 
      for indx in range(1000):
          _, x_eval, w, b = sess.run([train_op, x, weights, biases])
          
          xs.append(x_eval)
          ws.append(w)
          bs.append(b)

          # print("x = {}".format(x_eval))

      eng.put_var('xs', xs)
      eng.put_var('ws', ws)
      eng.put_var('bs', bs)



# --------------------------------------------------- #
# DENSE CLASS
# --------------------------------------------------- #



# --------------------------------------------------- #
# CONSTANTS CLASS
# --------------------------------------------------- #
class constants:
  MATLAB_SESSION  = 'desktop'


# --------------------------------------------------- #
# MAIN ENTRY POINT
# --------------------------------------------------- #

if __name__ == "__main__":
  args = sys.argv

  if args[1] == 'train' :
    train()
  else :
    pass
    