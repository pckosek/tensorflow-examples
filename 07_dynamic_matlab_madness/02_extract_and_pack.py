# --------------------------------------------------- #
# IMPORT STATEMENTS
# --------------------------------------------------- #
import tensorflow as tf
import numpy as np
import os
import extract_and_pack_features as f

# disable massive tensorflow start-log. 
#  only do this if you know the implications
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

MAX_TIME = 10


# --------------------------------------------------- #
# DEINE TRAINING DATA
# --------------------------------------------------- #
def get_data():
  x_train_batch, y_train_batch = f.inputs([f.TRAIN_TF_RECORDS_FILE], name="train")
  
  out = {
    'traX'  : x_train_batch,
    'traY'  : y_train_batch  
  }
  return out


# --------------------------------------------------- #
# DEINE TRAINING DATA
# --------------------------------------------------- #
def get_variables():
  # our big matrix
  init_state = tf.zeros([MAX_TIME, f.STRIDE_DIM], tf.float32)
  
  out = {
    'X' : tf.Variable(init_state, dtype=tf.float32)
  }
  return out


# --------------------------------------------------- #
# QUEUE OP
# --------------------------------------------------- #
def smoosh_fcn(queue_mat, new_data):
  _, bot = tf.split(queue_mat, [1, MAX_TIME-1], axis=0)
  out    = tf.concat([bot,new_data], axis=0 )

  reset  = tf.assign( queue_mat, out )
  return reset


# --------------------------------------------------- #
# MODEL
# --------------------------------------------------- #
def model(data, variables):
  queue_op = smoosh_fcn(variables['X'], data['traX'])

  return queue_op


# --------------------------------------------------- #
# TENSORFLOWWING!!!!
# --------------------------------------------------- #
def train():

  variables = get_variables()
  data      = get_data()

  queue_op  = model(data, variables)

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)

    # enable batch fetchers
    coord   = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for _ in range(5) :
      print( '------------' )
      print( sess.run( queue_op ) )

    coord.request_stop()
    coord.join(threads)


# --------------------------------------------------- #
# MAIN ENTRY POINT
# --------------------------------------------------- #
if __name__ == "__main__":
  train()