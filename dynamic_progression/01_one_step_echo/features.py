import tensorflow as tf
import numpy as np
import matlab.engine
import os

# disable that logging nonsense
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

TRAIN_TF_RECORDS_FILE = 'train_rnn.tf'
TEST_TF_RECORDS_FILE  = 'test_rnn.tf'


# NUMBER OF TIME STEPS OF INPUT, PER BATCH
INPUT_TIME_STEPS = 2
# size of the INPUT vector associated with a single time step 
INPUT_DIM = 1

# NUMBER OF TIME STEPS OF INPUT, PER BATCH
OUTPUT_TIME_STEPS = 2
# size of the OUTPUT vector associated with a single time step 
OUTPUT_DIM = 1

BATCH_SIZE = 1

# matlab session
SESSION_NAME = 'desktop'



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

def create_tfrecords(source, target, tf_records_file) :

  writer = tf.python_io.TFRecordWriter(tf_records_file)

  # IT'S CRITICAL THAT THIS STRUCTURE MATCHES THE DATA STRUCTURE
  # YOU ARE TRYING TO SAVE!!!
  for indx in range( source.shape[0] ):
    thisX  = source[indx].reshape( [INPUT_TIME_STEPS*INPUT_DIM] )
    thisY  = target[indx].reshape( [OUTPUT_TIME_STEPS*OUTPUT_DIM] )

    example = tf.train.Example(features=tf.train.Features(feature={
      'x': _float_list_feature(thisX),
      'y': _float_list_feature(thisY),
    }))
    writer.write(example.SerializeToString())

  writer.close()

# --------------------------------------------------- #
# FUNCTIONS FOR READING TF_RECORDS DATA
# --------------------------------------------------- #

# this first method grabs the data from the file
def read_and_decode(filename_queue, shuffle=False):
  reader = tf.TFRecordReader()

  _, serialized_example = reader.read(filename_queue)

  features = tf.parse_single_example(
      serialized_example,
      features={
          'x': tf.FixedLenFeature( [INPUT_TIME_STEPS*INPUT_DIM], tf.float32),
          'y': tf.FixedLenFeature( [OUTPUT_TIME_STEPS*OUTPUT_DIM], tf.float32),
      })

  x_cast = tf.cast(features['x'], tf.float32)
  y_cast = tf.cast(features['y'], tf.float32)

  x = tf.reshape(x_cast, [INPUT_TIME_STEPS, INPUT_DIM] )
  y = tf.reshape(y_cast, [OUTPUT_TIME_STEPS, OUTPUT_DIM] )

  # BATCH OPTIONALLY SHUFFLED DATA
  if shuffle==True :
    xb, yb = tf.train.shuffle_batch(
      [x, y],
      batch_size=BATCH_SIZE,
      capacity=100,
      min_after_dequeue=20)
  elif shuffle==False :
    xb, yb = tf.train.batch(
      [x, y], 
      batch_size=BATCH_SIZE)

  return xb, yb


# FUNCTION CALLED BY THE TENSORFLOW FETCH OPERATION
def inputs(filenames, shuffle=False, name=""):
  filename_queue = tf.train.string_input_producer(filenames)
  x, y = read_and_decode(filename_queue, shuffle)

  # GIVE THE INPUTS NAMES
  x = tf.identity(x, name="x_batch_{}".format(name))
  y = tf.identity(y, name="y_batch_{}".format(name))

  return x, y

# --------------------------------------------------- #
# OPERATIONS
# --------------------------------------------------- #

def write():
  # connect to matlab
  eng = matlab.engine.connect_matlab(SESSION_NAME)
  
  # grab sequence from MATLAB
  trainX = np.asarray( eng.workspace['trainX'] )
  trainY = np.asarray( eng.workspace['trainY'] )
  
  testX = np.asarray( eng.workspace['testX'] )
  testY = np.asarray( eng.workspace['testY'] )

  create_tfrecords( trainX, trainY, TRAIN_TF_RECORDS_FILE )
  create_tfrecords( testX, testY, TEST_TF_RECORDS_FILE )
    

def read():
  
  tr_x, tr_y = inputs( [TRAIN_TF_RECORDS_FILE] )

  with tf.Session() as sess:
    # enable batch fetchers
    coord   = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
  
    for n in range(4):
      print( sess.run( [tr_x, tr_y] ) )


    coord.request_stop()
    coord.join(threads)


if __name__ == "__main__":
  write()
  read()