#! /usr/bin/python
# www.github.com/andpol5/credit-classifier
#########################################################
# Train a linear regression  to predict on the german credit data
import time
import signal
import sys

import tensorflow as tf
import numpy as np
from numpy import genfromtxt

from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

def random_batch(dataset, batch_size):
    sample = dataset[np.random.choice(dataset.shape[0], batch_size, replace=False),:]
    last_col_index = dataset.shape[1]-2
    x = sample[:,0:last_col_index]
    y = sample[:,last_col_index:last_col_index+2]
    return (x, y)

# Constants
learning_rate = 1e-3
num_classes = 2
batch_size = 100
num_k_folds = 10

false_neg_cost = 1.0
false_pos_cost = 5.0

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

# The softmax function will make probabilties of Good vs Bad score at the output
y_ = tf.nn.softmax(tf.matmul(x, W) + b)

y = tf.placeholder(tf.float32, [None, num_classes])

# Training:
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

# Use 10-Fold cross validation to find the avg validation accuracy and
# confusion matrix values
kf = KFold(rows, n_folds=num_k_folds)
fold_counter = 1
val_conmats = []
val_precisions = []
val_recalls = []
val_f_scores = []

for train_indices, val_indices in kf:
    # split the data into train and validation
    train_dataset = dataset[train_indices,:]
    val_dataset = dataset[val_indices,:]

    for i in range(3000):
      batch_xs, batch_ys = random_batch(train_dataset, batch_size)
      sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

    # Get the ground truth values for this k-fold
    yT = np.argmax(val_dataset[:, 59:61], axis=1)

    # Evaluate the predicted values
    y_p = tf.argmax(y_,1)
    yP = sess.run(y_p, feed_dict={x: val_dataset[:, 0:59], y: val_dataset[:, 59:61]})

    # Metrics and confusion matrix
    [precision, recall, f_score, _] = precision_recall_fscore_support(yT, yP, average='macro')
    print("Validation k-fold #%d - precision: %f, recallL: %f, f-score: %f" % (
           fold_counter, precision, recall, f_score))
    conmat = confusion_matrix(yT, yP)
    print("Confusion matrix:")
    print("Good | Bad Credit")
    print conmat

    val_precisions.append(precision)
    val_recalls.append(recall)
    val_f_scores.append(f_score)
    val_conmats.append(conmat)
    fold_counter = fold_counter + 1

print("\nAveraging the 10-fold results:")
print("Validation precision - mean: %f, stddev: %f" % (
       np.mean(val_precisions), np.std(val_precisions)))
print("Validation recall - mean: %f, stddev: %f" % (
       np.mean(val_recalls), np.std(val_recalls)))
print("Validation f-score - mean: %f, stddev: %f" % (
       np.mean(val_f_scores), np.std(val_f_scores)))
print("Confusion matrix:")
print("Good | Bad Credit")
print (sum(val_conmats)).astype(float) / num_k_folds

