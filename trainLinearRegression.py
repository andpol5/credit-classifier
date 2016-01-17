#! /usr/bin/python
# Train a neural network to predict on the german credit data
import time
import signal
import sys

import tensorflow as tf
import numpy as np
from numpy import genfromtxt

def random_batch(dataset, batch_size):
    sample = dataset[np.random.choice(dataset.shape[0], batch_size, replace=False),:]
    last_col_index = dataset.shape[1]-2
    x = sample[:,0:last_col_index]
    y = sample[:,last_col_index:last_col_index+2]
    return (x, y)

# Constants
learning_rate = 1e-4
num_classes = 2
batch_size = 100

# Read the entire dataset (assumed to have already been preprocessed)
dataset = genfromtxt('newData.csv', delimiter=',')
rows, cols = dataset.shape

# assume the last 2 columns are the label
x_width = cols-2

# Placeholder values
x = tf.placeholder(tf.float32, [None, x_width])

# Try linear regression as first attempt
W = tf.Variable(tf.zeros([x_width, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# Training
y_ = tf.placeholder(tf.float32, [None, num_classes])
cross_entropy =  -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-15, 1.0)))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
init = tf.initialize_all_variables()

# Create session
sess = tf.InteractiveSession()
sess.run(init)
for i in range(5000):
  batch_xs, batch_ys = random_batch(dataset, batch_size)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print('Accuracy: %f' % sess.run(accuracy, feed_dict={x: dataset[:,0:59], y_: dataset[:,59:61]}))
