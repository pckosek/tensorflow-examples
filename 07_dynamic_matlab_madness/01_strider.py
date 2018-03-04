# --------------------------------------------------- #
# SLICING AND CONCATING
# --------------------------------------------------- #

import tensorflow as tf
import numpy as np
import os


# disable massive tensorflow start-log. 
#  only do this if you know the implications
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

X =  np.array( [ [1,2], [3,4], [4,5], [5, 6], [6, 7] ] , dtype=np.float32)

foo = tf.constant( X, dtype=tf.float32 )

top, bot = tf.split(foo, [1, 4], axis=0)

bar = tf.constant([99,100], dtype=tf.float32)
bar = tf.reshape(bar, [1, 2])

new = tf.concat( [bot,bar], axis=0 )


with tf.Session() as sess:

  for _ in range(1) :
    print( sess.run( top ) )
    print( '------------' )
    print( sess.run( bot ) )
    print( '------------' )
    print( sess.run( new ) )
