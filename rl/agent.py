import numpy as np
import tensorflow as tf


class Agent:
    def __init__(self, state_shape):
        self.state_in = tf.placeholder(shape=[None, state_shape], dtype=tf.float32)


class RandomAgent(Agent):
    """Agent th"""
    def __init__(self, state_shape, action_shape):
        super(RandomAgent, self).__init__(state_shape)
        self.output = [tf.constant(np.ones(action_shape)/action_shape)]
        self.chosen_action = tf.random_uniform([], maxval=action_shape, dtype=tf.int32)


class Reinforce(Agent):
    """Based on https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb."""
    def __init__(self, lr, state_shape, action_shape, hidden_sizes):
        super(Reinforce, self).__init__(state_shape)
        # Policy network
        hidden = self.state_in
        self.hidden = []
        for h_size in hidden_sizes:
            hidden = tf.contrib.slim.fully_connected(hidden, h_size,
                                                     biases_initializer=None,
                                                     activation_fn=tf.nn.relu)
            self.hidden.append(hidden)
        self.output = tf.contrib.slim.fully_connected(hidden, action_shape,
                                                      activation_fn=tf.nn.softmax,
                                                      biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        # Training
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = (tf.range(0, tf.shape(self.output)[0])
                        * tf.shape(self.output)[1] + self.action_holder)
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))
