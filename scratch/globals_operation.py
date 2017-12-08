import tensorflow as tf

def run_example():
	global_step = tf.Variable(0, name='global_step', trainable=False)

	x      = tf.Variable(0, dtype=tf.float32)
	labels = tf.constant(1.0, dtype=tf.float32)
	loss   = tf.reduce_sum(tf.square(x-labels))

	train_op = tf.train.AdamOptimizer(learning_rate=1.5).minimize(loss, global_step=global_step)	

	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


	with tf.Session() as sess:

		sess.run(init_op)

		for indx in range(100):
			sess.run(train_op)
			glob_step_eval = sess.run(global_step)
			x_eval = sess.run(x)

			print("global_step = {}, x = {}\n".format(glob_step_eval, x_eval))


if __name__ == '__main__':
    run_example()