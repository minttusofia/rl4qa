import numpy as np
import tensorflow as tf


class Agent:
    def __init__(self, state_shape):
        self.state_in = tf.placeholder(shape=[None, state_shape], dtype=tf.float32)


class RandomAgent(Agent):
    """Agent that picks actions in the action space uniformly at random."""
    def __init__(self, state_shape, action_shape):
        super(RandomAgent, self).__init__(state_shape)
        self.output = [tf.constant(np.ones(action_shape)/action_shape)]


class Reinforce(Agent):
    """Based on https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb."""
    def __init__(self, lr, state_shape, action_shape, hidden_sizes, entropy_w=0.):
        super(Reinforce, self).__init__(state_shape)
        # Policy network
        hidden = self.state_in
        self.hidden = []
        for h_size in hidden_sizes:
            hidden = tf.contrib.slim.fully_connected(hidden, h_size,
                                                     biases_initializer=None,
                                                     activation_fn=tf.nn.relu)
            self.hidden.append(hidden)
        self.softmax_output = tf.contrib.slim.fully_connected(hidden, action_shape,
                                                              activation_fn=tf.nn.softmax,
                                                              biases_initializer=None)
        # To avoid arithmetic overflow
        output = self.softmax_output + 1e-15
        # Renormalise per timestep
        # [timestep, action]/[timestep, 1]
        self.output = output/tf.reshape(tf.reduce_sum(output, axis=1), [-1, 1])

        # Training
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.entropy = -tf.reduce_sum(self.output * tf.log(self.output))

        # Indexes point to probabilities of actions taken:
        # timestep * num_actions + a_t for each timestep in episode
        self.indexes = (tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1]
                        + self.action_holder)
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.pg_loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)
        if entropy_w != 0.:
            self.ent_loss = -entropy_w * self.entropy
        else:  # avoid NaN entropy at convergence
            self.ent_loss = tf.constant(0.)
        self.loss = self.pg_loss + self.ent_loss

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))
