# coding:utf-8

# import tensorflow as tf
# tf.enable_eager_execution()

# labels = tf.constant([1, 1, 1], dtype=tf.int32)
# output = tf.constant([0.2, 0, 0], dtype=tf.float32)
# loss = tf.losses.mean_squared_error(labels, output)
# print(tf.sign(output))

# import random

# a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# b = ['a', 'b', 'c']

# ab = list(zip(a ,b))
# random.shuffle(ab)
# a, b = zip(*ab)
# print(a, b)


remove = lambda l : [i for i in l if i > 5]
a = [5, 6, 10]
print(remove(a))