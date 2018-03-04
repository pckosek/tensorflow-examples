# --------------------------------------------------- #
# IMPORT STATEMENTS
# --------------------------------------------------- #
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn 
import os
import extract_and_pack_features as f
import matlab.engine

# NEW INFO
# http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/


# disable massive tensorflow start-log. 
#  only do this if you know the implications
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# set up a place to save the model
saver_dir  = os.getcwd()
ckpt_path  = os.path.join(saver_dir, 'model.ckpt')
saver_path = os.path.join(ckpt_path, 'saver')


MAX_BACKPROPOGATION = 400

HIDDEN_DIM = 25

SESSION_NAME = 'desktop'

# AdamOptimizer => 10b == bad

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
  init_state  = tf.zeros([MAX_BACKPROPOGATION, f.STRIDE_DIM], tf.float32)
  
  out = {
    'gs'  : tf.Variable(0, name='global_step', trainable=False),
    'X' : tf.Variable(init_state, dtype=tf.float32), 
    'b_1' : tf.Variable(tf.random_normal(shape=[f.OUTPUT_DIM]), dtype=tf.float32),
    'w_1' : tf.Variable(tf.random_normal(shape=[HIDDEN_DIM, f.OUTPUT_DIM]), dtype=tf.float32)
  }
  return out


# --------------------------------------------------- #
# QUEUE OP
# --------------------------------------------------- #
def smoosh_fcn(queue_mat, new_data):
  _, bot = tf.split(queue_mat, [1, MAX_BACKPROPOGATION-1], axis=0)
  out    = tf.concat([bot,new_data], axis=0 )

  reset  = tf.assign( queue_mat, out )
  return reset


# --------------------------------------------------- #
# FUNCTIONS FOR TUPLE-BASED RNNs
# --------------------------------------------------- #
# (https://stackoverflow.com/questions/37969065/tensorflow-best-way-to-save-state-in-rnns/41240243)
def get_state_variables(batch_size, cell):
  # For each layer, get the initial state and make a variable out of it
  # to enable updating its value.
  state_variables = []
  for state_c, state_h in cell.zero_state(batch_size, tf.float32):
    state_variables.append(tf.contrib.rnn.LSTMStateTuple(
      tf.Variable(state_c, trainable=False),
      tf.Variable(state_h, trainable=False)))
  # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
  return tuple(state_variables)


def get_state_update_op(state_variables, new_states):
  # Add an operation to update the train states with the last state tensors
  update_ops = []
  for state_variable, new_state in zip(state_variables, new_states):
    # Assign the new state to the state variables on this layer
    update_ops.extend(
      [state_variable[0].assign(new_state[0]), state_variable[1].assign(new_state[1])]
    )
  # Return a tuple in order to combine all update_ops into a single operation.
  # The tuple's actual value should not be used.
  return tf.tuple(update_ops)

 
def get_state_reset_op(state_variables, cell, batch_size):
  # Return an operation to set each variable in a list of LSTMStateTuples to zero
  zero_states = cell.zero_state(batch_size, tf.float32)
  return get_state_update_op(state_variables, zero_states)
 
 
# --------------------------------------------------- #
# LOG HELPER FUNCTION
# --------------------------------------------------- #
def shape_log(tensor):
  print("{} has shape: {}".format(
    tensor.op.name,
    tensor.get_shape().as_list()))


# --------------------------------------------------- #
# MODEL
# --------------------------------------------------- #
def model(data, variables, op='train'):
  # TODO: reconcile agains this article
  # https://medium.com/@erikhallstrm/using-the-tensorflow-lstm-api-3-7-5f2b97ca6b73

  if (op=='train'):
    # feed_data = data['traX']
    dropout_pkeep = 0.5
  else :
    # feed_data = data['tstX']
    dropout_pkeep = 1.0

  NLAYERS = 1

  cells = [rnn.BasicLSTMCell(HIDDEN_DIM) for _ in range(NLAYERS)]
  # cells = [rnn.LSTMCell(HIDDEN_DIM) for _ in range(NLAYERS)]
  # cells = [rnn.GRUCell(NUM_NEURONS_1) for _ in range(NLAYERS)]
  # "naive dropout" implementation
  dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=dropout_pkeep) for cell in cells]
  multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=True)
  multicell = rnn.DropoutWrapper(multicell, output_keep_prob=dropout_pkeep) 

  states = get_state_variables(f.BATCH_SIZE, multicell)

  queue_op   = smoosh_fcn(variables['X'], data['traX'])
  reshape_op = tf.reshape( queue_op, [1, MAX_BACKPROPOGATION, f.STRIDE_DIM ], name='reshape_op')

  shape_log(reshape_op)

  outputs, H = tf.nn.dynamic_rnn(multicell, 
    reshape_op,
    initial_state=states,
    dtype=tf.float32)

  print( '-=-=-=-=-=-=-=-' )
  print( outputs )
  print( outputs[:,-1] )

  # STATE OPS. UPDATE | RESET
  update_op = get_state_update_op(states, H)
  reset_state_op = get_state_reset_op(states, multicell, f.BATCH_SIZE)

  # FULLY CONNECTED LAYER --1-- REDUCTION
  # 20 => 1
  mult_op = tf.matmul( 
   outputs[:,-1], 
   variables['w_1'], name='fc_1_mult')
  add_op  = tf.add( 
    mult_op, 
    variables['b_1'], name='fc_1_add' )

  logits = tf.identity(add_op, name='logits')

  # return logits, H, update_op, reset_state_op
  return logits, queue_op


# --------------------------------------------------- #
# TENSORFLOWWING!!!!
# --------------------------------------------------- #
def train(train_steps):

  variables = get_variables()
  data      = get_data()

  # logits
  logits, queue_op  = model(data, variables)

  # cost/loss
  loss = tf.reduce_sum(tf.square(logits - data['traY']))  # sum of the squares
  
  # optimizer
  # optimizer = tf.train.GradientDescentOptimizer(0.05)
  optimizer = tf.train.AdamOptimizer(0.05)
  train     = optimizer.minimize(loss, global_step=variables['gs'])

  #setup saver
  saver = tf.train.Saver(max_to_keep=3)

  # connect to matlab
  eng = matlab.engine.connect_matlab(SESSION_NAME)

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)

    # enable batch fetchers
    coord   = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # possibly restore model
    ckpt = tf.train.get_checkpoint_state(saver_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('saver restored')

    losses = []

    for i in range(train_steps) :
      _, step_loss = sess.run([train, loss] )
      losses.append( step_loss )

      # post an update
      if ((i+1)%250 == 0):
          print("[step: {}] loss: {}".format(i, step_loss))
          eng.workspace['losses'] = matlab.double( losses ) 


    coord.request_stop()
    coord.join(threads)

    # save the saver
    saver.save(sess, ckpt_path, global_step=variables['gs'])


# --------------------------------------------------- #
# TEST OPERATION
# --------------------------------------------------- #
def test(test_steps):
  variables = get_variables()
  data      = get_data()

  # logits
  logits, queue_op  = model(data, variables)
  
  #setup saver
  saver = tf.train.Saver(max_to_keep=3)

  # connect to matlab
  eng = matlab.engine.connect_matlab(SESSION_NAME)

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)

    # enable batch fetchers
    coord   = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # possibly restore model
    ckpt = tf.train.get_checkpoint_state(saver_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('saver restored')

    # output holders
    y_tests = []
    y_preds = []

    for i in range(test_steps) :

      # post an update
      if ((i+1)%250 == 0):
          print("[step: {}] ".format(i))

      # # RESET STATE IF FLAG IS PRESENT 
      # f_eval = sess.run( [data['traF']] )
      # if ( f_eval[0][0] == 1 ):
      #     sess.run( [reset_state_op] )
      
      y_test, test_predict = sess.run( [data['traY'], logits] )

      y_tests.append( y_test[0][0] )
      y_preds.append( test_predict[0][0] )

    # print( y_tests )
    # print('----------')
    # print( y_preds )

    eng.workspace['y_tests'] = matlab.double( y_tests ) 
    eng.workspace['y_preds'] = matlab.double( y_preds ) 


    coord.request_stop()
    coord.join(threads)


# --------------------------------------------------- #
# MAIN ENTRY POINT
# --------------------------------------------------- #
if __name__ == "__main__":
  train(7500)
  # test(1250)