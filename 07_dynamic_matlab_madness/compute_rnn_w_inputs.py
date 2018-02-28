# --------------------------------------------------- #
# IMPORT STATEMENTS
#
# ORIGINALLY SOURCED FROM 
# https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-12-5-rnn_stock_prediction.py
# 
# source use data of size
#  (716, 15, 5)    => each of the 716 :) inputs has 15 time steps with 5 points each 
#  (716, 1)        => each of the 716 :) outputs has 1 point
# 
# --------------------------------------------------- #
import tensorflow as tf
from tensorflow.contrib import rnn 
import numpy as np
import os
import matlab.engine
import features as f
import sys

# disable massive tensorflow start-log. 
#  only do this if you know the implications
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# set up a place to save the model
saver_dir  = os.getcwd()
ckpt_path  = os.path.join(saver_dir, 'model.ckpt')
saver_path = os.path.join(ckpt_path, 'saver')

# matlab session
SESSION_NAME = 'desktop'

# adjust the batch size from the features
f.BATCH_SIZE = 400

# number of steps to train for
TRAIN_STEPS = 2000

# set the number of hidden rnn states
HIDDEN_DIM = 10


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
# DEINE MODEL
# --------------------------------------------------- #
def model(data, variables, op='train'):

  if (op=='train'):
    feed_data = data['traX']
  else :
    feed_data = data['tstX']

  NLAYERS = 2
  dropout_pkeep = 1.0

  cells = [rnn.BasicLSTMCell(HIDDEN_DIM) for _ in range(NLAYERS)]
  # cells = [rnn.LSTMCell(HIDDEN_DIM) for _ in range(NLAYERS)]
  # cells = [rnn.GRUCell(NUM_NEURONS_1) for _ in range(NLAYERS)]
  # "naive dropout" implementation
  dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=dropout_pkeep) for cell in cells]
  multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=True)
  multicell = rnn.DropoutWrapper(multicell, output_keep_prob=dropout_pkeep) 

  states = get_state_variables(f.BATCH_SIZE, multicell)

  print('---------------------------------------------------------------------')
  print( states )
  # print( cells[0].get_shape().as_list() )
  # print( cells[1].get_shape().as_list() )
  shape_log( feed_data )

  outputs, H = tf.nn.dynamic_rnn(multicell, 
    feed_data,
    initial_state=states,
    dtype=tf.float32)

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

  return logits, H, update_op, reset_state_op
  # return logits, H, update_op

  # print( outputs.get_shape().as_list() )
# print( (outputs[:,-1]).get_shape().as_list() )
# [None, 15, 20] => [None, seq_length, hidden_dim]
# [None, 20]     => [None, hidden_dim]
# 
# which is to say we are only using hidden dim in
# mx + b


# --------------------------------------------------- #
# DEINE VARIABLES
# --------------------------------------------------- #
def get_variables():
  out = {
    'gs'  : tf.Variable(0, name='global_step', trainable=False),
    'b_1' : tf.Variable(tf.random_normal(shape=[f.OUTPUT_DIM]), dtype=tf.float32),
    'w_1' : tf.Variable(tf.random_normal(shape=[HIDDEN_DIM, f.OUTPUT_DIM]), dtype=tf.float32),
  }
  return out

  # 'b_1' : tf.Variable(tf.random_normal(shape=[HIDDEN_DIM]), dtype=tf.float32, name='bias_1'),
  # 'w_1' : tf.Variable(tf.random_normal(shape=[f.INPUT_DIM, HIDDEN_DIM]), dtype=tf.float32, name='weight_1'),

  # 'b_2' : tf.Variable(tf.random_normal(shape=[f.OUTPUT_DIM]), dtype=tf.float32, name='bias_2'),
  # 'w_2' : tf.Variable(tf.random_normal(shape=[HIDDEN_DIM, f.OUTPUT_DIM]), dtype=tf.float32, name='weight_2')


# --------------------------------------------------- #
# DEINE TRAINING DATA
# --------------------------------------------------- #
def get_data():
  x_train_batch, y_train_batch, f_train_batch = f.inputs([f.TRAIN_TF_RECORDS_FILE], name="train")
  x_test_batch,  y_test_batch,  f_test_batch  = f.inputs([f.TEST_TF_RECORDS_FILE], name="test")

  out = {
    'traX'  : x_train_batch,
    'traY'  : y_train_batch,
    'traF'  : f_train_batch,

    'tstX'  : x_test_batch,
    'tstY'  : y_test_batch,
    'tstF'  : f_test_batch
  }
  return out

# --------------------------------------------------- #
# DEINE PLACEHOLDERS
# --------------------------------------------------- #
def get_placeholders():
  out = {
    'X'  : tf.placeholder(tf.float32, [None, f.SEQ_LENGTH, f.DATA_DIM]),
    'Y'  : tf.placeholder(tf.float32, [None, 1])
  }
  return out

# 'Hin' : tf.placeholder(tf.float32, [None, 2*hidden_dim], name='Hin')  
# 'Hin' : tf.placeholder(tf.float32, [hidden_dim, 2*hidden_dim], name='Hin')  # worked before transition to stored tuple

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

    learning_rate = 0.01
    iterations    = 750

    data         = get_data()
    placeholders = get_placeholders()
    variables    = get_variables()

    # logits, update_op, reset_state_op = model(data, variables)
    logits, H, update_op, reset_state_op = model(data, variables, op='train')

    print('======================================')
    shape_log( data['traX'] )
    shape_log( logits )
    print( H )

    # cost/loss
    loss = tf.reduce_sum(tf.square(logits - data['traY']))  # sum of the squares
    
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train     = optimizer.minimize(loss, global_step=variables['gs'])

    #setup saver
    saver = tf.train.Saver(max_to_keep=3)

    # connect to matlab
    eng = matlab.engine.connect_matlab(SESSION_NAME)
        
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # enable batch fetchers
        coord   = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # possibly restore model
        ckpt = tf.train.get_checkpoint_state(saver_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('saver restored')

        # Training step
        for i in range(TRAIN_STEPS):
            f_eval = sess.run( [data['traF']] )

            # RESET STATE IF FLAG IS PRESENT 
            if ( f_eval[0][0] == 1 ):
                sess.run( [reset_state_op] )

            # _, step_loss, ostate = sess.run([train, loss, update_op], feed_dict={} )
            _, step_loss, ostate = sess.run([train, loss, update_op] )

            # post an update
            if ((i*f.BATCH_SIZE)%1600 == 0):
                # this will set the MATLAB variables
                hout               = sess.run( H )
                eng.workspace['H'] = matlab.double( np.asarray(hout).tolist() )
                eng.updateMatlab()

            # post an update
            if ((i+1)%250 == 0):
                print("[step: {}] loss: {}".format(i, step_loss))

        # save the saver
        saver.save(sess, ckpt_path, global_step=variables['gs'])


def test():

    # EVALUATE NETWORK

    data         = get_data()
    placeholders = get_placeholders()
    variables    = get_variables()

    logits, H, update_op, reset_state_op = model(data, variables, op='test')

    #setup saver
    saver = tf.train.Saver(max_to_keep=3)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
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

        # Evaluate op
        for i in range(10):
            
            y_test, test_predict = sess.run( [data['traY'], logits] )

            y_tests.append( y_test )
            y_preds.append( test_predict )


        y_tests = np.asarray(y_tests)
        y_preds = np.asarray(y_preds)

        # connect to matlab
        eng = matlab.engine.connect_matlab(SESSION_NAME)
        
        # this will set the MATLAB variables
        eng.workspace['testY'] = matlab.double( y_tests.tolist() ) 
        eng.workspace['test_predict'] = matlab.double( y_preds.tolist() ) 

        coord.request_stop()
        coord.join(threads)

    # within MATLAB, call this to plot
    # >>plot( [testY, test_predict], '+-' )

if __name__ == "__main__":
  test()
  # train()






  # args = sys.argv
  # print( args )
  # if args[1] is 'test' :
  #   print( '---testing---')
  #   test()
  # elif args[1] is 'train' :
  #   print( '---training---')
  #   train()
  # else :
  #   print( 'please specify desired operation ( <<test>> or <<train>> )' )