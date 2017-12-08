import tensorflow as tf

def run_example():

    tf_x    = tf.Variable([0.0, 1.0], dtype=tf.float32)

    reshape_op = tf.reshape(tf_x, [2, 1])

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        print( sess.run(reshape_op) )


if __name__ == '__main__':
    run_example()