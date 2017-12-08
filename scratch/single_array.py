import tensorflow as tf

def run_example():

    tf_x    = tf.Variable([0.0, 1.0], dtype=tf.float32)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        print( sess.run(tf_x) )


if __name__ == '__main__':
    run_example()