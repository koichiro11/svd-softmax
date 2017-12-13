# -*- coding: utf-8 -*-
"""
SVD-Softmax: Fast Softmax Approximation on Large Vocabulary Neural Networks
http://papers.nips.cc/paper/7130-svd-softmax-fast-softmax-approximation-on-large-vocabulary-neural-networks.pdf

implemented in Tensorflow by Koichiro Tamura(http://koichirotamura.com/)
"""

import tensorflow as tf
import math


def svd_softmax(dec, tgt_vocab_size, hidden_units, window_size=2*5, num_full_view=2**11):
    """
    svd-softmax 
    :param dec: A Tensor [batch_size, seq_length, hidden_units], decoder output
    :param tgt_vocab_size: int, num of vocabulary
    :param hidden_units: int, num of hidden units
    :param window_size: int, width of preview window W( hidden_units/ 8 is recommended)
    :param num_full_view: int, num of full-view size
    :return: A Tensor [batch_size, seq_length, tgt_vocab_size], output after softmax approximation
    """

    with tf.variable_scope("logits", reuse=True):
        weights = tf.Variable(
            tf.truncated_normal([tgt_vocab_size, hidden_units],
                                stddev=1.0 / math.sqrt(hidden_units)), name="output_weight")
        biases = tf.Variable(tf.zeros([tgt_vocab_size]), name="output_bias")
        dec_output = tf.reshape(dec, [-1, hidden_units])     # [batch_size*T_q, hidden]

        # svd-softmax
        _s, U, V = tf.svd(weights, full_matrices=False)
        B = tf.matmul(U, tf.diag(_s))

        _h = tf.einsum('ij,aj->ai', tf.transpose(V), dec_output)  # [batch_size*T_q, hidden]
        _z = tf.add(tf.einsum('ij,aj->ai', B[:, :window_size], _h[:, :window_size]), biases)  # [batch_size*T_q, voc]

        top_k = tf.nn.top_k(_z, k=tgt_vocab_size)
        _indices, values = top_k.indices, top_k.values  # [batch_size*T_q, N]

        _z = tf.add(tf.squeeze(tf.matmul(tf.gather(B, _indices[:, :num_full_view]), tf.expand_dims(_h, axis=-1))), tf.gather(biases, _indices[:, :num_full_view]))  # [N*T_q, N]
        _z = tf.concat([_z, values[:, num_full_view:]], axis=-1)
        _z = tf.map_fn(lambda x: tf.gather(x[0], tf.invert_permutation(x[1])), (_z, _indices), dtype=(tf.float32))      # [batch_size*T_q, voc]
        _z = tf.exp(_z)
        Z = tf.expand_dims(tf.reduce_sum(_z, axis=-1), axis=1)      # [batch_size*T_q, 1]
        logits = _z / Z

        return tf.reshape(logits, [-1, tf.shape(dec)[1], tgt_vocab_size])
