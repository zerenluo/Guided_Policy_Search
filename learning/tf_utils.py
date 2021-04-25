import numpy as np
import tensorflow as tf
from PythonLinearNonlinearControl.envs.make_envs import make_env


class ILNetwork(object):
    def __init__(self, params):
        self.sess = tf.Session()
        # build net and initialize variables when instantiate this class
        self.env = make_env(params)
        self.build_net()
        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        # create input
        self.input_r = tf.placeholder(dtype=tf.float32, shape=[None, self.env.config['state_size']], name='state')
        self.output_r = tf.placeholder(dtype=tf.float32, shape=[None, self.env.config['input_size']], name='action')

        # create variables
        with tf.variable_scope('controller_net'):
            n_l1, n_l2, n_l3, w_initializer, b_initializer = 64, 128, 64,\
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

            with tf.variable_scope('l_input'):
                w0 = tf.get_variable('w0', [self.env.config['state_size'], n_l1], initializer=w_initializer)
                b0 = tf.get_variable('b0', [1, n_l1], initializer=b_initializer)
                l0 = tf.nn.tanh(tf.matmul(self.input_r, w0) + b0)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [n_l1, n_l2], initializer=w_initializer)
                b1 = tf.get_variable('b1', [1, n_l2], initializer=b_initializer)
                l1 = tf.nn.tanh(tf.matmul(l0, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l2, n_l3], initializer=w_initializer)
                b2 = tf.get_variable('b2', [1, n_l3], initializer=b_initializer)
                l2 = tf.nn.tanh(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l_output'):
                w3 = tf.get_variable('w3', [n_l3, self.env.config['input_size']], initializer=w_initializer)
                b3 = tf.get_variable('b3', [1, self.env.config['input_size']], initializer=b_initializer)
                self.output_pred = tf.matmul(l2, w3) + b3

        # create loss
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.output_r, self.output_pred))

        # create optimizer
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)  # learning rate could be set in the first bracket

    def forward(self, inputs):
        output_pred = self.sess.run(self.output_pred, feed_dict={self.input_r: inputs} )
        return output_pred

    def learn(self, input_batch, output_batch):
        # run the optimizer and get the loss
        _, loss_run = self.sess.run([self.train_op, self.loss], feed_dict={self.input_r: input_batch, self.output_r: output_batch})

        return loss_run