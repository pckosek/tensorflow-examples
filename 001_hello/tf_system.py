import tensorflow as tf

x = tf.constant([0, 1., 2, .5], dtype=tf.float32, name="x")

with tf.Session() as sess:
    out = sess.run(x)
    print(out)
