#! /usr/bin/python
# Train a neural network to predict on the german credit data
import time
import signal
import sys

import tensorflow as tf
import numpy as np
from numpy import genfromtxt

from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

def random_batch(dataset, batch_size):
    sample = dataset[np.random.choice(dataset.shape[0], batch_size, replace=False),:]
    last_col_index = dataset.shape[1]-2
    x = sample[:,0:last_col_index]
    y = sample[:,last_col_index:last_col_index+2]
    return (x, y)

# Tensorflow convinience functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, mean=0.001, stddev=0.3)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Constants
learning_rate = 1e-4
num_classes = 2
batch_size = 100
dropout = 0.5

false_neg_cost = 1.0
false_pos_cost = 5.0

# Read the entire dataset (assumed to have already been preprocessed)
dataset = genfromtxt('newData.csv', delimiter=',')
rows, cols = dataset.shape

# assume the last 2 columns are the label
x_width = cols-2

# Placeholder values
x = tf.placeholder(tf.float32, [None, x_width])

# Neural network with 2 hidden layers

# Fully connected layer 1:
w_fc1 = weight_variable([x_width, 512])                   # weights
b_fc1 = bias_variable([512])                          # biases
h_fc1 = tf.nn.relu(tf.matmul(x, w_fc1) + b_fc1)       # activation
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)          # dropout

# Fully connected layer 2:
w_fc2 = weight_variable([512, 512])
b_fc2 = bias_variable([512])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# Readout layer
w_fc3 = weight_variable([512, num_classes])
b_fc3 = bias_variable([num_classes])

# The softmax function will make probabilties of Good vs Bad score at the output
y_ = tf.nn.softmax(tf.matmul(h_fc2_drop, w_fc3) + b_fc3)
y = tf.placeholder(tf.float32, [None, num_classes])

# Training
# Different loss functions:
# cross_entropy =  -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-15, 1.0)))

# Mofified mean square error function which uses a cost of 5 for false positives
# and a cost of one for false negatives
square_diff = tf.square(y - y_)
good, bad = tf.split(1, 2, square_diff)
costwise_loss = false_neg_cost*tf.reduce_sum(good) + false_pos_cost*tf.reduce_sum(bad)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(costwise_loss)

init = tf.initialize_all_variables()

# Create session
sess = tf.InteractiveSession()
sess.run(init)
for i in range(5000):
  batch_xs, batch_ys = random_batch(dataset, batch_size)
  sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

y_p = tf.argmax(y_,1)
accuracyD, yP = sess.run([accuracy, y_p], feed_dict={x: dataset[:,0:59], y: dataset[:,59:61], keep_prob: 1.0})
print("Accuracy: %f" % accuracyD)

# Get the confusion matrix
yT = np.argmax(dataset[:,59:61], axis=1)
conmat = confusion_matrix(yT, yP)
print("Confusion matrix:")
print("Good | Bad Credit")
print conmat
