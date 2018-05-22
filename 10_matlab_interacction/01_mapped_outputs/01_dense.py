# --------------------------------------------------- #
# IMPORT STATEMENTS
# --------------------------------------------------- #
import tensorflow as tf
from tensorflow.contrib import rnn 
import sys
import os
import numpy as np
import matlab_ops as mo
import utils as u

# disable tensorflow logging stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# --------------------------------------------------- #
# THE MODEL
# --------------------------------------------------- #
def model(X, c) :
  cells     = [
    rnn.LSTMCell(num_units=c.NUM_UNITS)
    for _ in range(c.NUM_LAYERS)]
  multicell = rnn.MultiRNNCell(cells, state_is_tuple=True)
 
  reshape_op = tf.reshape( X, [1, 100, 1], name='reshape_op')

  outputs, H = tf.nn.dynamic_rnn(multicell, 
    reshape_op,
    dtype=tf.float32)

  u.shape_log(outputs)

  # outputs has shape => [batch_size, max_time, num_units]
  # reshape it to     => [batch_size*max_time, num_units]
  # apply a fully connected layer with shared weight and bias
  # that reshapes to  => [batch_size*max_time, n_outputs]


  reshape_op = tf.reshape( outputs, [-1, c.NUM_UNITS] )
  unit_kern  = tf.Variable( tf.random_normal([1, 1]), dtype=tf.float32 )
  unit_bias  = tf.Variable( tf.random_normal([1, 1]), dtype=tf.float32 )
  kern       = tf.matmul(tf.ones([c.NUM_UNITS, 1],dtype=tf.float32), unit_kern)
  bias       = tf.matmul(tf.ones([100, 1],dtype=tf.float32), unit_bias)

  mult_op    = tf.matmul(reshape_op, kern, name='multop')
  dense_op   = tf.add( mult_op, bias, name='addop')

  u.shape_log(dense_op)
  
  logits     = tf.reshape(dense_op, [100])
  u.shape_log(logits)

  return logits


# --------------------------------------------------- #
# TRAINING METHOD
# --------------------------------------------------- #
def train():
  c = constants()

  # connect to matlab
  eng = mo.matlab_connection(c.MATLAB_SESSION)

  # matlab initialization...
  # [b,a] = butter(7, .2);
  # x = [zeros(1,10), ones(1,90)];
  # y = filter(b,a,x);

  mx = eng.get_var('x')
  my = eng.get_var('y')

  x      = tf.constant(mx, dtype=tf.float32)

  logits = model(x, c)

  labels = tf.constant(my, dtype=tf.float32)
  labels = tf.reshape( labels, [100])

  loss   = tf.reduce_mean(tf.square(logits-labels))

  train_op = tf.train.AdamOptimizer(learning_rate=0.05).minimize(loss)
  init_op  = tf.global_variables_initializer()

  with tf.Session() as sess:

      # initialize global variables
      sess.run(init_op)

      logi = []

      # run the training operation 
      for indx in range(750):
          _, log = sess.run([train_op, logits])
          logi.append(log)

          # post an update
          if ((indx+1)%250 == 0):
              print("[step: {}]".format(indx))
              
      eng.put_var('logits', logi)

# --------------------------------------------------- #
# CONSTANTS CLASS
# --------------------------------------------------- #
class constants():
  MATLAB_SESSION  = 'desktop'
  NUM_LAYERS = 5
  NUM_UNITS  = 20

# --------------------------------------------------- #
# MAIN ENTRY POINT
# --------------------------------------------------- #

if __name__ == "__main__":
  args = sys.argv

  if args[1] == 'train' :
    train()
  else :
    pass
    