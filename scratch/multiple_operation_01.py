import tensorflow as tf

def run_example():

    x  			= tf.constant([0, 1, 2, 3, 4, 5], dtype=tf.float32)

    reshape_op  = tf.reshape(x, [3, 2])
    add_op      = tf.add(reshape_op, 10.0)

    with tf.Session() as sess:
        print( sess.run(add_op) )


if __name__ == '__main__':
    run_example()