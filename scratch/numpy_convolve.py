import tensorflow as tf
import numpy as np

def run_example():

    x = [0, 1, 0, 0]
    y = [1, 2, 3]

    z = np.convolve(x, y, 'full')

    print(z)

    # tf_x    = tf.Variable(1.0, dtype=tf.float32)

    # init_op = tf.group(tf.global_variables_initializer(),
    #                    tf.local_variables_initializer())

    # with tf.Session() as sess:
    #     sess.run(init_op)

    #     print( sess.run([tf_x]) )


if __name__ == '__main__':
    # createRecords()
    # verify_tfecords()
    # restore_and_dump()
    run_example()
    # run_from_matlab()