import tensorflow as tf

def run_example():

    x           = tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float32)

    reshape_op  = tf.reshape(x, [3, 2])
    softmax_op  = tf.softmax(reshape_op)

    with tf.Session() as sess:

        res = sess.run(softmax_op)
        print("Result 1 = \n{}\n".format(res))


if __name__ == '__main__':
    run_example()