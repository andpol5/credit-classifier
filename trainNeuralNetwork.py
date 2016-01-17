#! /usr/bin/python
# Train a neural network to predict on the german credit data
import time
import signal
import sys

import tensorflow as tf
import numpy as np
from numpy import genfromtxt

def weight_variable(shape):
  initial = tf.truncated_normal(shape, mean=0.001, stddev=0.3)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Constants
learning_rate = 1e-4

# Read the entire dataset (assumed to have already been preprocessed)
dataset = genfromtxt('newData.csv', delimiter=',')
import ipdb; ipdb.set_trace()


# Placeholder values
x = tf.placeholder(tf.float32, [None, 784])

# Try linear regression as first attempt
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# Training
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
init = tf.initialize_all_variables()

# Create session
sess = tf.InteractiveSession()
sess.run(init)
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
