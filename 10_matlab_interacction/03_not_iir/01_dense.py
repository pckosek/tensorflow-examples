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
 
  reshape_op = tf.reshape( X, [-1, c.MAX_TIME, 1], name='reshape_op')

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
  bias       = tf.matmul(tf.ones([c.MAX_TIME, 1],dtype=tf.float32), unit_bias)

  mult_op    = tf.matmul(reshape_op, kern, name='multop')
  dense_op   = tf.add( mult_op, bias, name='addop')

  u.shape_log(dense_op)
  
  logits     = tf.reshape(dense_op, [c.MAX_TIME])
  u.shape_log(logits)

  return logits



# --------------------------------------------------- #
# TRAINING METHOD
# --------------------------------------------------- #
def train():
  c = constants()
  p = params()

  # connect to matlab
  eng = mo.matlab_connection(c.MATLAB_SESSION)

  # matlab initialization...
  # [b,a] = butter(7, .2);
  # x = [zeros(1,10), ones(1,90)];
  # y = filter(b,a,x);

  mx = eng.get_var('x')
  my = eng.get_var('y')

  x  = tf.constant(mx, dtype=tf.float32)

  logits = model(x, c)

  labels = tf.constant(my, dtype=tf.float32)
  labels = tf.reshape( labels, [c.MAX_TIME])

  loss   = tf.reduce_mean(tf.square(logits-labels))

  train_op = tf.train.AdamOptimizer(learning_rate=0.05).minimize(loss, global_step=p.global_step)
  init_op  = tf.global_variables_initializer()


  with tf.Session() as sess:

      # initialize global variables
      sess.run(init_op)

      #setup saver
      saver = u.saver_ops(sess=sess, max_to_keep=3, global_step=p.global_step)

      logi = []

      # run the training operation 
      for indx in range(750):
          _, log = sess.run([train_op, logits])
          logi.append(log)

          u.display_update(indx, 250)
              
      eng.put_var('logits', logi)

      saver.save()

 
# --------------------------------------------------- #
# TEST METHOD
# --------------------------------------------------- #
def test():
  c = constants()
  p = params()

  # connect to matlab
  eng = mo.matlab_connection(c.MATLAB_SESSION)

  # matlab initialization...
  # [b,a] = butter(7, .2);
  # x = [zeros(1,10), ones(1,90)];
  # y = filter(b,a,x);

  mx = eng.get_var('s')
  my = eng.get_var('t')

  x  = tf.constant(mx, dtype=tf.float32)

  logits = model(x, c)

  labels = tf.constant(my, dtype=tf.float32)
  labels = tf.reshape( labels, [c.MAX_TIME])

  init_op  = tf.global_variables_initializer()


  with tf.Session() as sess:

      # initialize global variables
      sess.run(init_op)

      #setup saver
      saver = u.saver_ops(sess=sess, max_to_keep=3, global_step=p.global_step)

      logi = []
      labs = []

      # run the training operation 
      for indx in range(1):
          log, lab = sess.run([logits, labels])
          logi.append(log)
          labs.append(lab)

          u.display_update(indx, 250)
              
      eng.put_var('logits', logi)
      eng.put_var('labels', labs)

      saver.save()


# --------------------------------------------------- #
# CONSTANTS CLASS
# --------------------------------------------------- #
class constants():
  MATLAB_SESSION  = 'desktop'
  NUM_LAYERS = 5
  NUM_UNITS  = 20
  MAX_TIME   = 100


# --------------------------------------------------- #
# PARAMS CLASS
# --------------------------------------------------- #
class params():
  global_step = tf.Variable(0, name='global_step', trainable=False)

# --------------------------------------------------- #
# TRAINING METHOD
# --------------------------------------------------- #

if __name__ == "__main__":
  args = sys.argv

  if args[1] == 'train' :
    train()
  else :
    test()
    