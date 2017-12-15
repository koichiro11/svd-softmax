# -*- coding: utf-8 -*-
"""
SVD-Softmax: Fast Softmax Approximation on Large Vocabulary Neural Networks
http://papers.nips.cc/paper/7130-svd-softmax-fast-softmax-approximation-on-large-vocabulary-neural-networks.pdf
implemented in Tensorflow by Koichiro Tamura(http://koichirotamura.com/)
"""

import tensorflow as tf
import math

class SVDSoftmax(object):
    """svd-softmax class"""

    def __init__(self, tgt_vocab_size, hidden_units, window_size=2 ** 5, num_full_view=2 ** 11):
        """
        initialize SVD
        :param tgt_vocab_size: int, num of vocabulary
        :param hidden_units: int, num of hidden units
        :param window_size: int, width of preview window W( hidden_units/ 8 is recommended)
        :param num_full_view: int, num of full-view size
        :return: A Tensor [batch_size, seq_length, tgt_vocab_size], output after softmax approximation
        """

        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_units = hidden_units
        self.window_size = window_size
        self.num_full_view = num_full_view

        # tf.matmul(U, tf.diag(_s))
        self.B = tf.Variable(
            tf.truncated_normal([self.tgt_vocab_size, self.hidden_units],
                                stddev=1.0 / math.sqrt(hidden_units)), name="B_SVD", trainable=False)
        # transposed V
        self.V_t = tf.Variable(
            tf.truncated_normal([self.hidden_units, self.hidden_units],
                                stddev=1.0 / math.sqrt(hidden_units)), name="V_SVD", trainable=False)

    def update_params(self, weights):
        """
        update svd parameter B, V_t
        :param weights: output weight of softmax
        :return:
        """
        _s, U, V = tf.svd(weights, full_matrices=False)
        self.B.assign(tf.matmul(U, tf.diag(_s)))
        self.V_t.assign(tf.transpose(V))
        return

    def get_output(self, dec_output, biases):
        """
        get svd-softmax approximation
        :param dec: A Tensor [batch_size*seq_length, hidden_units], decoder output
        :param biases: A Tensor [tgt_vocab_size], output bias
        :return: A Tensor [batch_size*seq_length, tgt_vocab_size], output after softmax approximation
        """
        _h = tf.einsum('ij,aj->ai', self.V_t, dec_output)
        _z = tf.add(tf.einsum('ij,aj->ai', self.B[:, :self.window_size], _h[:, :self.window_size]), biases)

        top_k = tf.nn.top_k(_z, k=self.tgt_vocab_size)
        _indices, values = top_k.indices, top_k.values

        _z = tf.add(tf.squeeze(tf.matmul(tf.gather(self.B, _indices[:, :self.num_full_view]), tf.expand_dims(_h, axis=-1))),
                    tf.gather(biases, _indices[:, :self.num_full_view]))
        _z = tf.concat([_z, values[:, self.num_full_view:]], axis=-1)
        _z = tf.map_fn(lambda x: tf.gather(x[0], tf.invert_permutation(x[1])), (_z, _indices), dtype=tf.float32)
        _z = tf.exp(_z)
        Z = tf.expand_dims(tf.reduce_sum(_z, axis=-1), axis=1)
        logits = _z / Z

        return logits
