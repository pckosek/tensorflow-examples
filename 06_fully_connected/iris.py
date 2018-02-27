# --------------------------------------------------- #
# THIS IS A SIMPLE SANITY CHECKER FOR FULLY CONNECTED DIMENSION CHECKING
# THE IDEA IS THAT I CAN CHANGE THE BATCH SIZE AND VIEW OUTPUT DIMENSIONS
#
# THE DATA SET IS DRAWN FROM MATLAB'S FISHERIRIS - WHICH IS CLEARLY A 
# CLASSIFICATION PROBLEM, BUT I'M USING IT HERE AS A REGRESSION PROBLEM
#
# I DO THIS BECAUSE I LIKE TO SMILE
# --------------------------------------------------- #


# --------------------------------------------------- #
# IMPORT STATEMENTS
# --------------------------------------------------- #
import tensorflow as tf
from tensorflow.contrib import rnn 
import numpy as np
import os
import matlab.engine
import features as f

# disable tensorflow logging stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# matlab session
SESSION_NAME = 'desktop'

# set the batch size
f.BATCH_SIZE = 1;

HIDDEN_DIM = 10

# --------------------------------------------------- #
# DEINE VARIABLES
# --------------------------------------------------- #
def get_variables():
  out = {
    'gs'  : tf.Variable(0, name='global_step', trainable=False),
    'b_1' : tf.Variable(tf.random_normal(shape=[HIDDEN_DIM]), dtype=tf.float32, name='bias_1'),
    'w_1' : tf.Variable(tf.random_normal(shape=[f.INPUT_DIM, HIDDEN_DIM]), dtype=tf.float32, name='weight_1'),

    'b_2' : tf.Variable(tf.random_normal(shape=[f.OUTPUT_DIM]), dtype=tf.float32, name='bias_2'),
    'w_2' : tf.Variable(tf.random_normal(shape=[HIDDEN_DIM, f.OUTPUT_DIM]), dtype=tf.float32, name='weight_2')

  }
  return out


# --------------------------------------------------- #
# DEINE TRAINING DATA
# --------------------------------------------------- #
def get_data():
  x_train_batch, y_train_batch = f.inputs([f.TRAIN_TF_RECORDS_FILE])
  x_test_batch,  y_test_batch  = f.inputs([f.TEST_TF_RECORDS_FILE])

  out = {
    'traX'  : x_train_batch,
    'traY'  : y_train_batch,
    'tstX'  : x_test_batch,
    'tstY'  : y_test_batch
  }
  return out


# --------------------------------------------------- #
# DEINE MODEL
# --------------------------------------------------- #
def model(data, variables):

  # FULLY CONNECTED LAYER --1-- EXPANSION
  # 4 => 10
  mult_op = tf.matmul( 
    data['traX'], 
    variables['w_1'], name='fc_1_mult')
  
  add_op  = tf.add( 
    mult_op, 
    variables['b_1'], name='fc_1_add' )
  
  # FULLY CONNECTED LAYER --1-- REDUCTION
  # 10 => 1
  mult_op = tf.matmul( 
    add_op, 
    variables['w_2'], name='fc_2_mult')

  add_op  = tf.add( 
    mult_op, 
    variables['b_2'], name='fc_2_add' )

  logits = tf.identity(add_op, name='logits')
  return logits


# --------------------------------------------------- #
# LOG HELPER FUNCTION
# --------------------------------------------------- #
def shape_log(tensor):
  print("{} has shape: {}".format(
    tensor.op.name,
    tensor.get_shape().as_list()))


# --------------------------------------------------- #
# TENSORFLOWWING!!!
# --------------------------------------------------- #
def train():
  data      = get_data()
  variables = get_variables()

  logits    = model(data, variables)

  # loss 
  train_loss = tf.reduce_sum(tf.square(logits - data['traY'])) 

  # opt op
  opt      = tf.train.AdamOptimizer(0.05)
  train_op = opt.minimize(train_loss, global_step=variables['gs'])

  with tf.Session() as sess:
    # get the show started
    init = tf.global_variables_initializer()
    sess.run(init)

    # enable batch fetchers
    coord   = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    shape_log( data['traX'] )
    shape_log( data['traY'] )
    shape_log( logits )

    sess.run( [train_op] )
    print( sess.run([logits]) )


# --------------------------------------------------- #
# int main()
# --------------------------------------------------- #  
if __name__ == "__main__":
  train()
  # read()
  False