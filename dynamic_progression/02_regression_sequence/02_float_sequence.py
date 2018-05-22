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
import features as f

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
def get_data(constants):
  x_train_batch, y_train_batch = f.inputs(
    [f.TRAIN_TF_RECORDS_FILE],
    shuffle=constants['SHUFFLE'],
    name="train")
  x_test_batch, y_test_batch = f.inputs(
    [f.TEST_TF_RECORDS_FILE], 
    shuffle=constants['SHUFFLE'],
    name="test")
  
  out = {
    'traX'  : x_train_batch,
    'traY'  : y_train_batch, 
    'tstX'  : x_train_batch,
    'tstY'  : y_train_batch 
  }
  return out


# --------------------------------------------------- #
# DEINE TRAINING DATA
# --------------------------------------------------- #
def get_variables(constants):
  
  out = {
    'gs'  : tf.Variable(0, name='global_step', trainable=False),
    'w_1' : tf.Variable(tf.random_normal(
      [constants['HIDDEN_DIM'],constants['OUTPUT_DIM']]), name='w_1'),
    'b_1' : tf.Variable(tf.random_normal(
      [constants['OUTPUT_TIME_STEPS'],constants['OUTPUT_DIM']]), name='b_1')
  }
  return out


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
    feed_data = data['traX']
    dropout_pkeep = 0.5
    print('model set to train')
  else :
    feed_data = data['tstX']
    dropout_pkeep = 1.0
    print('model set to test')

  # PREP DATA FOR INPUT
  input_reshape_op = tf.reshape(feed_data, 
    [-1, constants['INPUT_TIME_STEPS'], constants['INPUT_DIM'] ], 
    name='input_reshape_op')

  # CREATE CELLS LSTMCell
  basic_cell = rnn.LSTMCell(
    num_units=constants['HIDDEN_DIM'],
    activation=tf.nn.tanh)

  rnn_output, states = tf.nn.dynamic_rnn(basic_cell,
    input_reshape_op,
    time_major=False,
    dtype=tf.float32)

  stacked_rnn_output = tf.reshape(rnn_output, 
    [-1, constants['HIDDEN_DIM']], name='stacked_rnn')

   # stacked_outputs    = tf.layers.dense(stacked_rnn_output,
   #  constants['OUTPUT_TIME_STEPS'], name='dense_output')

  # FULLY CONNECTED LAYER --1-- REDUCTION
  # HIDDEN_DIM * MAX_BACKPROPOGATION => MAX_BACKPROPOGATION
  mult_op = tf.matmul( 
   stacked_rnn_output, 
   variables['w_1'], name='fc_1_mult')

  add_op  = tf.add( 
    mult_op, 
    variables['b_1'], name='fc_1_add' )

  reshape_out = tf.reshape(
    add_op,
    [-1, constants['OUTPUT_TIME_STEPS'], constants['OUTPUT_DIM']] , name='reshout_2')

  print( '***********************' )
  shape_log(input_reshape_op)
  shape_log(stacked_rnn_output)
  shape_log(mult_op)
  shape_log(add_op)
  shape_log(reshape_out)

  logits = tf.reshape(reshape_out, [constants['OUTPUT_TIME_STEPS'] *constants['OUTPUT_DIM']] )
    # [-1, constants['OUTPUT_TIME_STEPS'], constants['OUTPUT_DIM']])
  labels = tf.reshape(data['traY'], [constants['OUTPUT_TIME_STEPS'] *constants['OUTPUT_DIM']] )
    # [-1, constants['OUTPUT_TIME_STEPS'], constants['OUTPUT_DIM']])

  # logits = tf.clip_by_value(logits, 0., 1.)

  logits = tf.identity(logits, name='logits')
  labels = tf.identity(labels, name='labels')

  shape_log(logits)
  shape_log(labels)
  
  return logits, labels
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
  data      = get_data(constants)

  # logits
  logits, labels = model(data, variables, constants, op='train')

  # L1 Loss function
  # loss = tf.reduce_sum(tf.abs(labels - logits))  # sum of the squares
  # L2 Loss function
  loss = tf.reduce_sum(tf.square(logits - labels))  # sum of the squares
  
  # https://stackoverflow.com/questions/35961216/
  reg_loss = loss + constants['LAMBDA_REG'] * (tf.nn.l2_loss(variables['w_1']) + tf.nn.l2_loss(variables['b_1']))

  # optimizer
  # optimizer = tf.train.GradientDescentOptimizer(0.01)
  optimizer = tf.train.AdamOptimizer(0.005)
  train     = optimizer.minimize(reg_loss, global_step=variables['gs'])

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
      _, step_loss, log, lab = sess.run([train, reg_loss, logits, labels] )
      losses.append( step_loss )

      # post an update
      if ((i+1)%251 == 0):
          print("[step: {}] loss: {}\nlogits:{}\nlabels: {}\n".format(i, step_loss, log, lab))
          eng.workspace['losses'] = matlab.double( losses ) 

          # CHECK EMERGENCY STOP
          should_stop = eng.should_stop()
          if should_stop:
            print('should_stop')
            break

      if step_loss < constants['LOSS_THRESHOLD'] :
        print( "stopping training; loss({}) < {}".format(step_loss,constants['LOSS_THRESHOLD']))
        eng.workspace['losses'] = matlab.double( losses ) 
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
  data      = get_data(constants)

  # logits
  logits, labels = model(data, variables, constants, op='test')
  
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

    w, b = sess.run([variables['w_1'], variables['b_1']])

    # SEEMS TO BE THE MOST ROBUST WAY OF TURNING INTO MATLAB DATA
    eng.workspace['y_tests'] = matlab.double( np.array(y_tests).tolist() ) 
    eng.workspace['y_preds'] = matlab.double( np.array(y_preds).tolist() ) 
    eng.workspace['w'] = matlab.double( np.array(w).tolist() ) 
    eng.workspace['b'] = matlab.double( np.array(b).tolist() ) 


    coord.request_stop()
    coord.join(threads)


# --------------------------------------------------- #
# CONSTANTS
# --------------------------------------------------- #
constants = {
  'INPUT_TIME_STEPS'  : 4,
  'INPUT_DIM'         : 1, 
  'OUTPUT_TIME_STEPS' : 4,
  'OUTPUT_DIM'        : 1, 
  'HIDDEN_DIM'        : 64,
  'SESSION_NAME'      : 'desktop',
  'LOSS_THRESHOLD'    : 1e-5,
  'SHUFFLE'           : False,
  'LAMBDA_REG'        : 5e-2
}

def update_constants(HIDDEN_DIM=None, SESSION_NAME=None):
  if HIDDEN_DIM is not None:
    constants['HIDDEN_DIM'] = HIDDEN_DIM


# --------------------------------------------------- #
# MAIN ENTRY POINT
# --------------------------------------------------- #
if __name__ == "__main__":
  args = sys.argv

  if len(args) > 2 :
    update_constants(HIDDEN_DIM=int(args[2]))
    # constants['MAX_BACKPROPOGATION'] = int(args[2])

  if args[1] == 'train' :
    train(8000)
  elif args[1] == 'test' :
    test(100)