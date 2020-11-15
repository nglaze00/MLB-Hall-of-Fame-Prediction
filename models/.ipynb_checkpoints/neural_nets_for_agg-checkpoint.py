from __future__ import division
from __future__ import print_function
import nn
import numpy as np
import pickle
import time
import tensorflow.compat.v1 as tf
from sklearn.metrics import classification_report
from scipy.special import softmax

tf.compat.v1.disable_eager_execution()


def construct_feed_dict(features, labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    return feed_dict


# Set random seed
# seed = 125
seed = 128
np.random.seed(seed)
tf.set_random_seed(seed)

# Settingss
## best: .001, 3 layers
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.1, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_string('save_name', './agg_nn/agg_nn_model', 'Path for saving model')
flags.DEFINE_integer('layers', 3, 'number of layers')

# Load the data
test_data = np.load("../data_ready/agg/X_test.npy")
train_data = np.load("../data_ready/agg/X_train.npy")
test_labels = 1-np.load("../data_ready/agg/y_test_1hot.npy")
train_labels = 1-np.load("../data_ready/agg/y_train_1hot.npy")

last_years_hof_idxs = np.load('../data_ready/test_last_years_hof_idxs.npy')
last_years_idxs = np.load('../data_ready/test_last_years_idxs.npy')

last_years = test_data[last_years_idxs]
last_years_labels = test_labels[last_years_idxs]
last_years_hof = test_data[last_years_hof_idxs]
last_years_hof_labels = test_labels[last_years_hof_idxs]

# print(last_years.shape, last_years_hof.shape, last_years_labels.shape)

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

# cost_val = []
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
          "train_acc=", "{:.5f}".format(outs[2]), "time=", "{:.5f}".format(time.time() - t), end=' ')
    _, full_hof_acc, _ = evaluate(last_years_hof, last_years_hof_labels, placeholders)
    _, full_all_acc, _ = evaluate(last_years, last_years_labels, placeholders)
    print('full career hof acc:', full_hof_acc, 'full career all acc:', full_all_acc)

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

feed_dict = construct_feed_dict(last_years, last_years_labels, placeholders)


preds = sess.run(model.predict(), feed_dict=feed_dict)

print(softmax(preds, axis=1)[:1000])

# print(np.sum(test_labels), np.sum(np.argmax(preds, axis=1)))
print(classification_report(np.argmax(last_years_labels, axis=1), np.argmax(preds, axis=1)))


feed_dict = construct_feed_dict(test_data, test_labels, placeholders)
preds = sess.run(model.predict(), feed_dict=feed_dict)
print(preds)
print(preds.mean(axis=0), preds.max(axis=0))
print(softmax(preds, axis=1))
raise Exception
np.save('agg_nn/probs.npy', 1 - softmax(preds, axis=1))
np.save('agg_nn/preds.npy', 1 - np.argmax(preds, axis=1))
print(sum(1 - np.argmax(preds, axis=1)))