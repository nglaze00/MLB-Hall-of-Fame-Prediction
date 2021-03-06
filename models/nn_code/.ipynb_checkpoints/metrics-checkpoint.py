import tensorflow.compat.v1 as tf


def softmax_cross_entropy(preds, labels):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    return tf.reduce_mean(loss)

def weighted_softmax_cross_entropy(preds, labels):
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=preds, labels=labels, pos_weight=0.25)
    return tf.reduce_mean(loss)


def accuracy(preds, labels):
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)


def mean_squared_error(preds, labels):
    loss = tf.losses.mean_squared_error(labels=labels, predictions=preds)
    return tf.reduce_mean(loss)
