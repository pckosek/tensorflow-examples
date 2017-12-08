import tensorflow as tf

def run_example():

    x           = tf.placeholder(tf.float32)

    reshape_op  = tf.reshape(x, [3, 2])
    add_op      = tf.add(reshape_op, 10.0)

    with tf.Session() as sess:
        res = sess.run(add_op, feed_dict={x : [10, 2, 3, 4, 5, 6]})
        print("Result 1 = \n{}\n".format(res))

        res = sess.run(add_op, feed_dict={x : [-1, -2, -3, -4, -5, -6]})
        print("Result 2 = \n{}\n".format(res))



if __name__ == '__main__':
    run_example()