from __future__ import division
from __future__ import print_function
import nn
import numpy as np
import time
import data
import utils
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt



# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_string('save_name', './mymodel.ckpt', 'Path for saving model')
test_data = np.load("test_numpy.npy")
train_data = np.load("train_numpy.npy")
test_labels = np.load("test_labels.npy")
train_labels = np.load("train_labels.npy")

