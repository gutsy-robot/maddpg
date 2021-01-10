import tensorflow as tf

a = tf.get_variable("A/a", initializer=tf.constant(4, shape=[2]))
b = tf.get_variable("B/b", initializer=tf.constant(5, shape=[3]))
init_op = tf.global_variables_initializer()

# with tf.Session() as sess:
#     # initialize all of the variables in the session
#     sess.run(init_op)
#     # run the session to get the value of the variable
#     a_out, b_out = sess.run([a, b])
#     print('a = ', a_out)
#     print('b = ', b_out)

print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="B"))
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="B")
                       + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="A"))

# saver = tf.train.Saver()
# for i, var in enumerate(saver._var_list):
#     print('Var {}: {}'.format(i, var))
#
# with tf.Session() as sess:
#     # initialize all of the variables in the session
#     sess.run(init_op)
#
#     # save the variable in the disk
#     saved_path = saver.save(sess, './saved_variable')
#     print('model saved in {}'.format(saved_path))

#
with tf.Session() as sess:
    # restore the saved vairable
    saver.restore(sess, './saved_variable')
    # print the loaded variable
    a_out, b_out = sess.run([a, b])
    print('a = ', a_out)
    print('b = ', b_out)