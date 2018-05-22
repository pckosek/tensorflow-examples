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
import features as f

# disable tensorflow logging stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# --------------------------------------------------- #
# THE MODEL
# --------------------------------------------------- #
def model(X, c) :
  cells     = [
    rnn.LSTMCell(num_units=c.NUM_UNITS)
    for _ in range(c.NUM_LAYERS)
  ]
  multicell = rnn.MultiRNNCell(cells, state_is_tuple=True)
 
  reshape_op = tf.reshape( X, [-1, c.MAX_TIME, 2], name='reshape_op')

  outputs, H = tf.nn.dynamic_rnn(multicell, 
    reshape_op,
    dtype=tf.float32)

  u.shape_log(outputs)

  reshape_op = tf.reshape( outputs, [-1, c.NUM_UNITS] )
  u.shape_log(reshape_op)
  # dense_op   = u.distributed_dense(reshape_op, units=2, max_time=100)
  dense_op  = tf.layers.dense(reshape_op, 2, name='distributed__dense')
  u.shape_log(dense_op)


  w, b = u.get_layer_vars('distributed__dense')
 
  logits     = tf.reshape(dense_op, [c.MAX_TIME*f.OUTPUT_DIM])
  u.shape_log(logits)

  return logits


# --------------------------------------------------- #
# TRAINING METHOD
# --------------------------------------------------- #
def train():
  c = constants()
  p = params()

  x, y = f.inputs([f.TRAIN_TF_RECORDS_FILE], shuffle=True, name='inputs')
  u.shape_log(x)
  u.shape_log(y)

  logits = model(x, c)

  logits = tf.reshape( logits, [-1, c.MAX_TIME, f.OUTPUT_DIM])
  labels = tf.reshape( y, [-1, c.MAX_TIME, f.OUTPUT_DIM])

  loss   = tf.reduce_mean(tf.square(logits-labels))

  train_op = tf.train.AdamOptimizer(learning_rate=0.05).minimize(loss, global_step=p.global_step)
  init_op  = tf.global_variables_initializer()

  with tf.Session() as sess:

      # initialize global variables
      sess.run(init_op)
  
      # enable batch fetchers
      coord   = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      #setup saver
      saver = u.saver_ops(sess=sess, max_to_keep=3, global_step=p.global_step)

      eng = mo.matlab_connection(c.MATLAB_SESSION)

      # run the training operation 
      for indx in range(20000):
          _, log = sess.run([train_op, logits])
          # logi.append(log)

          u.display_update(indx, 25)
              
          if ((indx+1)%2000 == 0):
            saver.save()

          if ((indx)%5 == 0):
            # CHECK EMERGENCY STOP
            if eng.should_stop():
              print('stopping')
              break            

      # connect to matlab
      # eng = mo.matlab_connection(c.MATLAB_SESSION)
      # eng.put_var('logits', logi)

      saver.save()
      coord.request_stop()
      coord.join(threads)

 
# --------------------------------------------------- #
# TEST METHOD
# --------------------------------------------------- #
def test():
  c = constants()
  p = params()

  # connect to matlab
  eng = mo.matlab_connection(c.MATLAB_SESSION)

  x, y = f.inputs([f.TEST_TF_RECORDS_FILE], shuffle=True, name='inputs')
  u.shape_log(x)
  u.shape_log(y)

  logits = model(x, c)
  logits = tf.reshape( logits, [-1, c.MAX_TIME, f.OUTPUT_DIM])
  labels = tf.reshape( y, [-1, c.MAX_TIME, f.OUTPUT_DIM])

  init_op  = tf.global_variables_initializer()

  with tf.Session() as sess:

      # initialize global variables
      sess.run(init_op)

      # enable batch fetchers
      coord   = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      #setup saver
      saver = u.saver_ops(sess=sess, max_to_keep=3, global_step=p.global_step)

      logi = []
      labs = []

      # run the training operation 
      for indx in range(6):
          log, lab = sess.run([logits, labels])
          logi.append(log)
          labs.append(lab)

          u.display_update(indx, 250)

      eng.put_var('logits', logi)
      eng.put_var('labels', labs)

      coord.request_stop()
      coord.join(threads)


def dream():
  c = constants()
  p = params()

  # connect to matlab
  eng = mo.matlab_connection(c.MATLAB_SESSION)

  x = tf.placeholder(tf.float32, shape=(1,100,2))

  logits = model(x, c)
  logits = tf.reshape( logits, [-1, c.MAX_TIME, f.OUTPUT_DIM])

  init_op  = tf.global_variables_initializer()

  rand_state = np.random.uniform(-.005,.005,(1,100,2))

  t  = np.linspace(0,1,100)
  x1 = .65 + .25 * np.cos( 4.5*2.0*np.pi*t )
  x2 = .45 + .25 * np.cos( 6.2*2.0*np.pi*t )
  istate = np.asarray( [x1,x2] ) 
  istate = [np.transpose(istate)]
  istate = istate + rand_state

  with tf.Session() as sess:

      # initialize global variables
      sess.run(init_op)

      # enable batch fetchers
      coord   = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      #setup saver
      saver = u.saver_ops(sess=sess, max_to_keep=3, global_step=p.global_step)

      xs   = []
      logi = []

      # run the training operation 
      for indx in range(1200):
          feed_dict = { x : istate }
          ex, ostate = sess.run([x, logits], feed_dict=feed_dict)
          logi.append(ostate[0])
          xs.append(ex[0])
          istate = np.roll(istate, -1, axis=1)
          istate[-1, -1, :] = ostate[-1, -1, :]

          u.display_update(indx, 20)

      eng.put_var('logits', logi)
      eng.put_var('x', xs)

      # matlab command =>
      # >>plot( reshape(logits(1,:,:), [100,2]) )

      coord.request_stop()
      coord.join(threads)


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
  elif args[1] == 'test' :
    test()
  else :
    dream()
    