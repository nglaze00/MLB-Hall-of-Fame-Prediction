import pandas as pd
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()

import numpy as np
import layer
import metrics

flags = tf.app.flags
FLAGS = flags.FLAGS


class MLB:
    def __init__(self, placeholders, layers, input_dim, output_dim, **kwargs):

        self.vars = {}
        self.placeholders = placeholders
        self.layers = []
        self.activations = []
        self.num_layers = layers
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.opt_op = None

        self.build()

    def _build(self):

        self.layers.append(layer.Dense(input_dim=self.input_dim,
                                       output_dim=FLAGS.hidden1,
                                       placeholders=self.placeholders,
                                       act=tf.nn.relu,
                                       dropout=True))

        for _ in range(self.num_layers - 2):
            self.layers.append(layer.Dense(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden1,
                                           placeholders=self.placeholders,
                                           act=tf.nn.relu,
                                           dropout=True))

        self.layers.append(layer.Dense(input_dim=FLAGS.hidden1,
                                       output_dim=self.output_dim,
                                       placeholders=self.placeholders,
                                       act=tf.nn.relu,
                                       dropout=True))

    def build(self):
        """ Wrapper for _build() """

        self._build()

        # Build sequential layer model
        # Feed the values from the previous layer to the next layer
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # self.vars = {var.name: var for var in variables}
        print("variables =", tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op2 = self.optimizer.compute_gradients(self.loss, variables)[1]

        self.opt_op = self.optimizer.minimize(self.loss)

    def _loss(self):

        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
#         self.loss += metrics.softmax_cross_entropy(self.outputs, self.placeholders['labels'])
        self.loss += metrics.weighted_softmax_cross_entropy(self.outputs, self.placeholders['labels'])
        
    def _accuracy(self):
        self.accuracy = metrics.accuracy(self.outputs, self.placeholders['labels'])

    def predict(self):
        return self.outputs

