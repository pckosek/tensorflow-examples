import tensorflow as tf
import numpy as np
import matlab.engine
import os

# disable that logging nonsense
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

TRAIN_TF_RECORDS_FILE = 'train_rnn.tf'
TEST_TF_RECORDS_FILE  = 'test_rnn.tf'

# number of 'previous' input vectors packed into a single time step
SEQ_LENGTH = 15 # 7
# size of the INPUT vector associated with a single time step 
DATA_DIM = 5
# size of the OUTPUT vector associated with a single time step 
OUTPUT_DIM = 1

BATCH_SIZE = 100

# matlab session
SESSION_NAME = 'MATLAB_6192'



# --------------------------------------------------- #
# HELPER FUNCTIONS FOR TYPES AND LISTS OF TYPES
# (https://github.com/bgshih/seglink/blob/master/tool/create_datasets.py)
# these will be used in creating tfrecords
# --------------------------------------------------- #
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
def _int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# --------------------------------------------------- #
# FUNCTION FOR CREATING TF_RECORDS DATA
# --------------------------------------------------- #

def create_tfrecords(source, target, flags, tf_records_file) :

  writer = tf.python_io.TFRecordWriter(tf_records_file)

  # IT'S CRITICAL THAT THIS STRUCTURE MATCHES THE DATA STRUCTURE
  # YOU ARE TRYING TO SAVE!!!
  for indx in range( source.shape[0] ):
    thisX  = source[indx].reshape( [SEQ_LENGTH*DATA_DIM] )
    thisY  = target[indx].reshape( [OUTPUT_DIM] )
    thisF  = flags[indx].reshape( [OUTPUT_DIM] )

    example = tf.train.Example(features=tf.train.Features(feature={
      'x': _float_list_feature(thisX),
      'y': _float_list_feature(thisY),
      'f': _float_list_feature(thisF)
    }))
    writer.write(example.SerializeToString())

  writer.close()

# --------------------------------------------------- #
# FUNCTIONS FOR READING TF_RECORDS DATA
# --------------------------------------------------- #

# this first method grabs the data from the file
def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()

  _, serialized_example = reader.read(filename_queue)

  features = tf.parse_single_example(
      serialized_example,
      features={
          'x': tf.FixedLenFeature([SEQ_LENGTH*DATA_DIM], tf.float32),
          'y': tf.FixedLenFeature([OUTPUT_DIM], tf.float32),
          'f': tf.FixedLenFeature([OUTPUT_DIM], tf.float32)
      })

  x_cast = tf.cast(features['x'], tf.float32)
  y_cast = tf.cast(features['y'], tf.float32)
  f_cast = tf.cast(features['f'], tf.float32)

  x = tf.reshape(x_cast, [SEQ_LENGTH, DATA_DIM] )
  y = tf.reshape(y_cast, [OUTPUT_DIM] )
  f = tf.reshape(f_cast, [OUTPUT_DIM] )

  xb, yb, fb = tf.train.batch([x, y, f], batch_size=BATCH_SIZE)
  return xb, yb, fb


# FUNCTION CALLED BY THE TENSORFLOW FETCH OPERATION
def inputs(filenames, name=""):
  filename_queue = tf.train.string_input_producer(filenames)
  x, y, f = read_and_decode(filename_queue)

  # GIVE THE INPUTS NAMES
  x = tf.identity(x, name="x_batch_{}".format(name))
  y = tf.identity(y, name="y_batch_{}".format(name))
  f = tf.identity(f, name="f_batch_{}".format(name))

  return x, y, f

# --------------------------------------------------- #
# OPERATIONS
# --------------------------------------------------- #

def write():
  # connect to matlab
  eng = matlab.engine.connect_matlab(SESSION_NAME)
  
  # grab sequence from MATLAB
  trainX = np.asarray( eng.workspace['trainX'] )
  trainY = np.asarray( eng.workspace['trainY'] )

  testX  = np.asarray( eng.workspace['testX'] )
  testY  = np.asarray( eng.workspace['testY'] )

  trainI = np.asarray( eng.workspace['trainI'] )
  testI  = np.asarray( eng.workspace['testI'] )
  
  create_tfrecords( trainX, trainY, trainI, TRAIN_TF_RECORDS_FILE )
  create_tfrecords( testX, testY, testI, TEST_TF_RECORDS_FILE )
  # t = trainX.reshape( [800*15*5] )
  # r = t.reshape( (800,15,5) )
  
  # # this will set the MATLAB variables
  # eng.workspace['r'] = matlab.double( r.tolist() ) 
    
def read():
  
  d1, d2, d3 = inputs( [TRAIN_TF_RECORDS_FILE] )
  d4, d5, d6 = inputs( [TEST_TF_RECORDS_FILE] )

  with tf.Session() as sess:
    # enable batch fetchers
    coord   = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
  
    for n in range(4):
      lastval = sess.run([d3])
      print( lastval[0][0] )

    coord.request_stop()
    coord.join(threads)


if __name__ == "__main__":
  1
  # write()
  # read()