from __future__ import division
from __future__ import print_function
import nn
import numpy as np
import time
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()


def construct_feed_dict(features, labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    return feed_dict


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.1, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_string('save_name', './mymodel.ckpt', 'Path for saving model')
flags.DEFINE_integer('layers', 5, 'number of layers')

# Load the data
test_data = np.load("test_numpy.npy")
train_data = np.load("train_numpy.npy")
test_labels = np.load("test_labels.npy")
train_labels = np.load("train_labels.npy")

# Define placeholders
placeholders = {
    'features': tf.placeholder(tf.float32, shape=(None, train_data.shape[1])),
    'labels': tf.placeholder(tf.float32, shape=(None, train_labels.shape[1])),
    'dropout': tf.placeholder_with_default(0., shape=()),
}

# Create model
model_func = nn.MLB
model = model_func(placeholders, layers=FLAGS.layers, input_dim=train_data.shape[1], output_dim=train_labels.shape[1])

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, labels, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

cost_val = []
t_0 = time.time()

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(train_data, train_labels, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    # cost, acc, duration = evaluate(features, y_val, placeholders)
    # cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          # "val_loss=", "{:.5f}".format(cost),
          # "val_acc=", "{:.5f}".format(acc),
          "train_acc=", "{:.5f}".format(outs[2]), "time=", "{:.5f}".format(time.time() - t))

#    if epoch > 500 and cost_val[-1] > np.mean(cost_val[-(1000 + 1):-1]):
 #       print("Early stopping...")
  #      break

saver.save(sess, FLAGS.save_name)

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(test_data, test_labels, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

# Testing only last years hof
# Load the data
last_years_hof = np.load("last_years_hof.npy")
last_years_hof_labels = np.load("last_years_hof_labels.npy")

print("NOW LAST YEAR HOF")
test_cost, test_acc, test_duration = evaluate(last_years_hof, last_years_hof_labels, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

print('Total time:', time.time() - t_0)

## only hof
feed_dict = construct_feed_dict(last_years_hof, last_years_hof_labels, placeholders)
prediction = sess.run(model.predict(), feed_dict=feed_dict)


def prediction_to_accuracy(predictions):
    size = predictions.shape[0]

    return np.sum((np.argmax(predictions, axis=1) == 0).astype(int)) / size


print(np.argmax(prediction, axis=1))
print(np.mean(np.equal(np.argmax(last_years_hof_labels, axis=1), np.argmax(prediction, axis=1)).astype(float)))
