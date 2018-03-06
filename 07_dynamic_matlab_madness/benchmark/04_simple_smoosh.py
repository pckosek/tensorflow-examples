# --------------------------------------------------- #
# IMPORT STATEMENTS
# --------------------------------------------------- #
import tensorflow as tf
from tensorflow.contrib import rnn 
import numpy as np
import matlab.engine
import os
import sys
# --------------------------- #
import extract_and_pack_features as f

# NEW INFO
# http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/


# disable massive tensorflow start-log. 
#  only do this if you know the implications
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# set up a place to save the model
saver_dir  = os.getcwd()
ckpt_path  = os.path.join(saver_dir, 'model.ckpt')
saver_path = os.path.join(ckpt_path, 'saver')

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
def get_variables(constants):

  MAX_BACKPROPOGATION = constants['MAX_BACKPROPOGATION']
  GENERATED_SEQ_LEN   = constants['GENERATED_SEQ_LEN']
  HIDDEN_DIM          = constants['HIDDEN_DIM']

  # our big matrix
  init_state         = tf.zeros([MAX_BACKPROPOGATION, f.STRIDE_DIM], tf.float32)
  zero_target_state  = tf.zeros([GENERATED_SEQ_LEN, 1], tf.float32)
  
  out = {
    'gs'  : tf.Variable(0, name='global_step', trainable=False),
    'X'   : tf.Variable(init_state, dtype=tf.float32, name='X'), 
    'Y'   : tf.Variable(zero_target_state, dtype=tf.float32, trainable=False, name='Y'), 
    'b_1' : tf.Variable(tf.random_normal(shape=[MAX_BACKPROPOGATION, 1]), dtype=tf.float32, name='b_1'),
    'w_1' : tf.Variable(tf.random_normal(shape=[HIDDEN_DIM, 1]), dtype=tf.float32, name='w_1'),
    'b_2' : tf.Variable(tf.random_normal(shape=[GENERATED_SEQ_LEN]), dtype=tf.float32, name='b_2'),
    'w_2' : tf.Variable(tf.random_normal(shape=[MAX_BACKPROPOGATION, GENERATED_SEQ_LEN]), dtype=tf.float32, name='w_2')
  }
  return out


# --------------------------------------------------- #
# QUEUE OP
# --------------------------------------------------- #
def get_src_update_fcn(variables, data):
  smoosh_input  = smoosh_fcn(variables['X'], data['traX'])
  smoosh_target = smoosh_fcn(variables['Y'], data['traY'])
  return smoosh_input, smoosh_target


def smoosh_fcn(queue_mat, new_data):
  print(queue_mat.get_shape().as_list() )
  print(new_data.get_shape().as_list() )
  num_rows = queue_mat.get_shape().as_list()[0]
  _, bot   = tf.split(queue_mat, [1, num_rows-1], axis=0)
  out      = tf.concat([bot,new_data], axis=0 )

  reset    = tf.assign( queue_mat, out )
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
def model(data, variables, constants, op='train'):
  # TODO: reconcile agains this article
  # https://medium.com/@erikhallstrm/using-the-tensorflow-lstm-api-3-7-5f2b97ca6b73

  if (op=='train'):
    # feed_data = data['traX']
    dropout_pkeep = 0.5
    print('model set to train')
  else :
    # feed_data = data['tstX']
    dropout_pkeep = 1.0
    print('model set to test')


  # DATA QUEUEING
  update_X_op, update_Y_op = get_src_update_fcn(variables, data)

  # PREP DATA FOR INPUT
  prep_input_op = tf.reshape( update_X_op, [-1, constants['MAX_BACKPROPOGATION'], f.STRIDE_DIM ], name='reshape_op')

  # CREATE CELLS
  basic_cell = rnn.LSTMCell(
    num_units=constants['HIDDEN_DIM'],
    activation=tf.nn.relu)
  rnn_output, states = tf.nn.dynamic_rnn(basic_cell,
    prep_input_op,
    dtype=tf.float32)

  stacked_rnn_output = tf.reshape(rnn_output, [-1, constants['HIDDEN_DIM']])
  stacked_outputs    = tf.layers.dense(stacked_rnn_output, 1)

  shape_log(stacked_rnn_output)
  shape_log(stacked_outputs)

  logits = tf.reshape( stacked_outputs, [-1, constants['MAX_BACKPROPOGATION'], 1])
  labels = tf.reshape( update_Y_op, [1, constants['MAX_BACKPROPOGATION'], 1])

  # logits = tf.clip_by_value(logits, 0., 1.)
  shape_log(logits)
  shape_log(labels)
  
  return logits, labels, update_X_op, update_Y_op
  # NLAYERS = 2

  # cells = [rnn.LSTMCell(HIDDEN_DIM) for _ in range(NLAYERS)]
  # # cells = [rnn.GRUCell(NUM_NEURONS_1) for _ in range(NLAYERS)]
  # dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=dropout_pkeep) for cell in cells]
  # multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=True)
  # multicell = rnn.DropoutWrapper(multicell, output_keep_prob=dropout_pkeep) 

  # states = get_state_variables(f.BATCH_SIZE, multicell)


  # # reshape input data for RNN
  # # reshape_op   = tf.reshape( update_X_op, [1, MAX_BACKPROPOGATION, f.STRIDE_DIM ], name='reshape_op')

  # shape_log(reshape_op)

  # outputs, H = tf.nn.dynamic_rnn(multicell, 
  #   reshape_op,
  #   initial_state=states,
  #   dtype=tf.float32)

  # reshape_out = tf.reshape( outputs, [-1, HIDDEN_DIM] , name='reshout_1')

  # print( '-=-=-=-=-=-=-=-' )
  # print( outputs )
  # # print( outputs[:,-1] )
  # print( reshape_out )



  # # STATE OPS. UPDATE | RESET
  # update_op = get_state_update_op(states, H)
  # reset_state_op = get_state_reset_op(states, multicell, f.BATCH_SIZE)

  # # FULLY CONNECTED LAYER --1-- REDUCTION
  # # HIDDEN_DIM * MAX_BACKPROPOGATION => MAX_BACKPROPOGATION
  # mult_op = tf.matmul( 
  #  reshape_out, 
  #  variables['w_1'], name='fc_1_mult')
  # shape_log(mult_op)
  # add_op  = tf.add( 
  #   mult_op, 
  #   variables['b_1'], name='fc_1_add' )
  # shape_log(add_op)

  # reshape_out = tf.reshape( add_op, [-1, MAX_BACKPROPOGATION] , name='reshout_2')
  # print( '***********************' )
  # shape_log(reshape_out)


  # # FULLY CONNECTED LAYER --1-- REDUCTION
  # # MAX_BACKPROPOGATION => GENERATED_SEQ_LEN
  # mult_op = tf.matmul( 
  #  reshape_out, 
  #  variables['w_2'], name='fc_2_mult')
  # add_op  = tf.add( 
  #   mult_op, 
  #   variables['b_2'], name='fc_2_add' )

  # print( '*-*-*-*-*-*-*-*-*-*-*-*-*-*' )
  # shape_log( add_op )
  #  # outputs[:,-1], 

  # logits = tf.reshape(add_op, [1, GENERATED_SEQ_LEN])
  # labels = tf.reshape(update_Y_op, [1, GENERATED_SEQ_LEN])

  # logits = tf.identity(logits, name='logits')
  # labels = tf.identity(labels, name='labels')



  # print('-------------')
  # shape_log( logits)
  # shape_log( labels)

  # # return logits, H, update_op, reset_state_op
  # return logits, labels, update_X_op, update_Y_op


# --------------------------------------------------- #
# TENSORFLOWWING!!!!
# --------------------------------------------------- #
def train(train_steps):

  variables = get_variables(constants)
  data      = get_data()

  shape_log(variables['X'])
  shape_log(variables['Y'])

  # logits
  logits, labels, update_X_op, update_Y_op = model(data, variables, constants, op='train')

  # cost/loss
  loss = tf.reduce_sum(tf.square(logits - labels))  # sum of the squares
  
  # optimizer
  optimizer = tf.train.GradientDescentOptimizer(0.0005)
  # optimizer = tf.train.AdamOptimizer(0.0005)
  train     = optimizer.minimize(loss, global_step=variables['gs'])

  #setup saver
  saver = tf.train.Saver(max_to_keep=3)

  # connect to matlab
  eng = matlab.engine.connect_matlab(constants['SESSION_NAME'])

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

          # CHECK EMERGENCY STOP
          should_stop = eng.should_stop()
          if should_stop:
            print('should_stop')
            break


    coord.request_stop()
    coord.join(threads)

    # save the saver
    saver.save(sess, ckpt_path, global_step=variables['gs'])


# --------------------------------------------------- #
# TEST OPERATION
# --------------------------------------------------- #
def test(test_steps):
  variables = get_variables(constants)
  data      = get_data()

  # logits
  logits, labels, update_X_op, update_Y_op = model(data, variables, constants, op='test')
  
  #setup saver
  saver = tf.train.Saver(max_to_keep=3)

  # connect to matlab
  eng = matlab.engine.connect_matlab(constants['SESSION_NAME'])

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

      y_tests.append( y_test )
      y_preds.append( test_predict )

    # SEEMS TO BE THE MOST ROBUST WAY OF TURNING INTO MATLAB DATA
    eng.workspace['y_tests'] = matlab.double( np.array(y_tests).tolist() ) 
    eng.workspace['y_preds'] = matlab.double( np.array(y_preds).tolist() ) 
    eng.workspace['backprop'] = matlab.double( [constants['MAX_BACKPROPOGATION']] ) 
    f = eng.call_from_python()

    coord.request_stop()
    coord.join(threads)


# --------------------------------------------------- #
# CONSTANTS
# --------------------------------------------------- #
constants = {
  'MAX_BACKPROPOGATION'  : 30,
  'HIDDEN_DIM'           : 50, 
  'GENERATED_SEQ_LEN'    : 30,
  'SESSION_NAME'         : 'desktop'
}

def update_constants(MAX_BACKPROPOGATION=None, HIDDEN_DIM=None, GENERATED_SEQ_LEN=None,SESSION_NAME=None):
  if MAX_BACKPROPOGATION is not None:
    constants['MAX_BACKPROPOGATION'] = MAX_BACKPROPOGATION
    constants['GENERATED_SEQ_LEN'] = MAX_BACKPROPOGATION 


# --------------------------------------------------- #
# MAIN ENTRY POINT
# --------------------------------------------------- #
if __name__ == "__main__":
  args = sys.argv

  if len(args) > 2 :
    update_constants(MAX_BACKPROPOGATION=int(args[2]))
    # constants['MAX_BACKPROPOGATION'] = int(args[2])

  if args[1] == 'train' :
    train(3000)
  elif args[1] == 'test' :
    test(300)