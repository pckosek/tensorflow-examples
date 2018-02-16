
import tensorflow as tf
import numpy as np


# --------------------------------------------------- #
# HELPER FUNCTIONS FOR TYPES AND LISTS OF TYPES
# (https://github.com/bgshih/seglink/blob/master/tool/create_datasets.py)
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

def create_tfrecords(data, filename) :

  writer = tf.python_io.TFRecordWriter(filename)

  # IT'S CRITICAL THAT THIS STRUCTURE MATCHES THE DATA STRUCTURE
  # YOU ARE TRYING TO SAVE!!!
  for indx in range( data.shape[0] ):
    example = tf.train.Example(features=tf.train.Features(feature={
        'x': _float_list_feature(data[indx])    
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
          'x': tf.FixedLenFeature([3], tf.float32),
      })

  single_x = tf.cast(features['x'], tf.float32)

  # dequeue parameters : source (http://adventuresinmachinelearning.com/introduction-tensorflow-queuing/)
  batch_size = 4
  num_threads = 4
  min_after_dequeue = 100
  capacity = min_after_dequeue + (num_threads + 1) * batch_size

  x = tf.train.shuffle_batch([single_x], 
    batch_size=batch_size,
    num_threads=num_threads,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue)
  return x


# FUNCTION CALLED BY THE TENSORFLOW FETCH OPERATION
def inputs(file_names):
  filenames = file_names
  filename_queue = tf.train.string_input_producer(filenames)
  x = read_and_decode(filename_queue)
  return x


# --------------------------------------------------- #
# START THE TENSORFLOWWING!!!
# --------------------------------------------------- #


# create a 2-d array
foo = []
for indx in range(100) :
  foo.append( 10*indx + np.array([0, 1, 2]) )
  if (indx+1 == 50):
    foo = np.asarray( foo )
    foo = foo.astype(np.float32)
    create_tfrecords(foo, filename='example_06a.tf')
    foo = []

foo = np.asarray( foo )
foo = foo.astype(np.float32)
create_tfrecords(foo, filename='example_06b.tf')

# create an 'input' variable
d1 = inputs(['example_06a.tf'])
d2 = inputs(['example_06b.tf'])
d3 = inputs(['example_06a.tf','example_06b.tf'])

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session() as sess:
  sess.run(init_op)

  coord   = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for n in range(0, 3):
      retval_d1 = sess.run([d1])
      print("batch from d1: \n{}".format(retval_d1))
      retval_d2 = sess.run([d2])
      print("batch from d2: \n{}".format(retval_d2))
      retval_d3 = sess.run([d3])
      print("batch from d3: \n{}\n".format(retval_d3))

  coord.request_stop()
  coord.join(threads)