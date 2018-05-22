# --------------------------------------------------- #
# IMPORT STATEMENTS
# --------------------------------------------------- #
import tensorflow as tf
import sys
import os
import matlab.engine
import numpy as np

# disable tensorflow logging stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# --------------------------------------------------- #
# TRAINING METHOD
# --------------------------------------------------- #

def train():
  c = constants()

  some_example_random_number = 0.31415

  x      = tf.Variable(some_example_random_number, dtype=tf.float32)
  labels = tf.constant(1.0, dtype=tf.float32)
  loss   = tf.reduce_sum(tf.square(x-labels))

  train_op = tf.train.AdamOptimizer(learning_rate=0.5).minimize(loss)
  init_op  = tf.global_variables_initializer()

  # connect to matlab
  eng = matlab_connection(c.MATLAB_SESSION)

  with tf.Session() as sess:

      # initialize global variables
      sess.run(init_op)

      xs = []

      # run the training operation 
      for indx in range(1000):
          _, x_eval = sess.run([train_op, x])
          
          xs.append(x_eval)

          # print("x = {}".format(x_eval))

      eng.put_var('xs', xs)


# --------------------------------------------------- #
# MATLAB CONNECTION CLASS
# --------------------------------------------------- #
class matlab_connection:

  def __init__(self, session_name):
    self.session_name = session_name
    self.eng = matlab.engine.connect_matlab(session_name)

  def put_var(self, var_name, values) :
    self.eng.workspace[var_name] = matlab.double( np.asarray(values).tolist() ) 

    

# --------------------------------------------------- #
# CONSTANTS CLASS
# --------------------------------------------------- #
class constants():
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
    