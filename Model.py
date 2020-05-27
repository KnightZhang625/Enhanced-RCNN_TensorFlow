# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 26_May_2020
# TensorFlow Version for Enhanced-RCNN.
#
# For GOD I Trust.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import copy
import tensorflow as tf
import model_helper as _mh

class ERCNNModel(object):
  """The main model for Enhanced-RCNN."""
  def __init__(self,
               config,
               is_training,
               sent_A,
               sent_B,
               sent_length_A,
               sent_length_B,
               scope=None):
    """Constructor for Enhanced-RCNN Model.
    
    Args:
      config: Config object, hyparameters set.
      is_training: boolean, whether train or not.
      sent_A: tf.int32 Tensor, sentence A, expected shape: [batch, seq_length_A].
      sent_B: tf.int32 Tensor, sentence B, expected shape: [batch, seq_length_B].
      sent_length_A: tf.int32 Tensor, the length for the sentence A, expected shape: [batch].
      sent_length_B: tf.int32 Tensor, the length for the sentence B, expected shape: [batch].
    
    Attention:
      `sent_length_A` must be identical to the `sent_length_b`, why? See the `similarity_model` function,
      click close button if do not understand.
    """
    # config
    config = copy.deepcopy(config)

    # RNN
    self.vocab_size = config.vocab_size
    self.embedding_size = config.embedding_size
    self.num_layers = config.num_layers
    self.hidden_size = config.hidden_size
    self.forget_bias = config.forget_bias
    self.sent_length_A = sent_length_A

    # CNN
    self.kernel_size = config.kernel_size
    self.pool_size= config.pool_size

    self.initializer_range = config.initializer_range
    self.dropout = config.dropout
    if not is_training:
      self.dropout = 0.0
    
    if is_training:
      self.output = self.build(sent_A, sent_B, sent_length_A, sent_length_B)

  def build(self,
            sent_A,
            sent_B,
            sent_length_A,
            sent_length_B,
            scope=None):
    # RNN Encoder
    encoder_outputs_A = RNNEncoder(sent_A,
                                   sent_length_A,
                                   self.vocab_size,
                                   self.embedding_size,
                                   self.num_layers,
                                   self.hidden_size,
                                   self.forget_bias,
                                   self.dropout,
                                   self.initializer_range)
    encoder_outputs_B = RNNEncoder(sent_B,
                                   sent_length_B,
                                   self.vocab_size,
                                   self.embedding_size,
                                   self.num_layers,
                                   self.hidden_size,
                                   self.forget_bias,
                                   self.dropout,
                                   self.initializer_range)
    
    # CNN
    cnn_output_A = CNNExtractor(encoder_outputs_A,
                                self.kernel_size,
                                self.pool_size,
                                self.dropout,
                                self.initializer_range)
    cnn_output_B = CNNExtractor(encoder_outputs_B,
                                self.kernel_size,
                                self.pool_size,
                                self.dropout,
                                self.initializer_range)
    
    # Attention
    attention_A = AttentionLayer(encoder_outputs_A, encoder_outputs_B)
    attention_B = AttentionLayer(encoder_outputs_B, encoder_outputs_A)

    # Max and Mean on the concatenate of the encoder outputs and the attention outputs
    V_a = tf.concat((encoder_outputs_A, attention_A, encoder_outputs_A - attention_A, tf.multiply(encoder_outputs_A, attention_A)), axis=-1)
    V_b = tf.concat((encoder_outputs_B, attention_B, encoder_outputs_B - attention_B, tf.multiply(encoder_outputs_B, attention_B)), axis=-1)
    v_a_max = tf.reduce_max(V_a, axis=-1)
    v_a_avg = tf.reduce_mean(V_a, axis=-1)
    v_b_max = tf.reduce_max(V_b, axis=-1)
    v_b_avg = tf.reduce_mean(V_b, axis=-1)

    # concatenate the final output
    # (8*s_a -8)
    output_a = tf.concat((v_a_max, cnn_output_A, v_a_avg), axis=-1)
    # (8*s_b -8)
    output_b = tf.concat((v_b_max, cnn_output_B, v_b_avg), axis=-1)

    output = self.similarity_model(output_a, output_b)

    with tf.variable_scope('prediction'):
      layer_size = _mh.get_shape_list(output)[1] // 2
      output = tf.layers.dense(output,
                               layer_size,
                               activation=tf.nn.tanh,
                               name='layer_mid',
                               kernel_initializer=_mh.create_initializer(initializer_range=self.initializer_range))
      output = tf.layers.dense(output,
                               2,
                               activation=tf.nn.tanh,
                               name='layer_final',
                               kernel_initializer=_mh.create_initializer(initializer_range=self.initializer_range))
    
    return output

  def similarity_model(self, output_a, output_b):
    def model_func(input_a, input_b, scope):
      with tf.variable_scope(scope):
        shape = [(self.sent_length_A[0] * 8 - 8) * 4, self.sent_length_A[0] * 8 - 8]
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='w')
        b = tf.Variable(tf.truncated_normal([self.sent_length_A[0] * 8 - 8], stddev=0.1), name='b')
      input_concat = tf.concat((input_a, input_b, tf.multiply(input_a, input_b), input_a - input_b), axis=-1)

      output = tf.nn.tanh(tf.nn.bias_add(tf.matmul(input_concat, w), b))
      return output

    def build_gate(input_a, input_b, scope):
      with tf.variable_scope(scope):
        input_concat = tf.concat((input_a, input_b), axis=-1)
        w_g = tf.Variable(tf.truncated_normal([(self.sent_length_A[0] * 8 - 8) * 2, self.sent_length_A[0] * 8 - 8]), name='w')
        # (b, seq_length)
        gate = tf.sigmoid(tf.matmul(input_concat, w_g))
      return gate

    # (b, s)
    m_oa_ob = model_func(output_a, output_b, scope='m_a')
    m_ob_oa = model_func(output_b, output_a, scope='m_b')

    # State Function
    gate_a = build_gate(output_a, output_b, scope='gate_a')
    gate_b = build_gate(output_b, output_a, scope='gate_b')

    output_a = tf.multiply(gate_a, m_oa_ob) + (1 - gate_a) * output_a
    output_b = tf.multiply(gate_b, m_ob_oa) + (1 - gate_b) * output_b
    output_concat = tf.concat((output_a, output_b), axis=-1)

    return output_concat
    
def RNNEncoder(input_text,
               input_length,
               vocab_size,
               embedding_size,
               num_layers,
               hidden_size,
               forget_bias,
               dropout,
               initializer_range,
               scope=None):
  # Embedding
  with tf.variable_scope('Embedding', reuse=tf.AUTO_REUSE):
    embedding_table = _mh.create_embedding(vocab_size,
                                            embedding_size,
                                            name='nmt_embedding',
                                            initializer_range=initializer_range)
    embedded_input = tf.nn.embedding_lookup(embedding_table, input_text)
  
  # RNN
  with tf.variable_scope('RNN', reuse=tf.AUTO_REUSE):
    assert_op = tf.assert_equal(num_layers % 2, 0)
    with tf.control_dependencies([assert_op]):
      num_bi_layers = int(num_layers / 2)
      num_bi_residual_layers = num_bi_layers - 1

      fw_cells = _mh.create_cell_list_for_RNN('gru',
                                              hidden_size,
                                              num_bi_layers,
                                              dropout,
                                              num_bi_residual_layers,
                                              forget_bias)

      bw_cells = _mh.create_cell_list_for_RNN('gru',
                                              hidden_size,
                                              num_bi_layers,
                                              dropout,
                                              num_bi_residual_layers,
                                              forget_bias)

      bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        fw_cells, bw_cells, embedded_input, dtype=tf.float32,
        sequence_length=input_length)
      
      encoder_outputs = tf.concat(bi_outputs, -1)

  return encoder_outputs
  
def CNNExtractor(inputs,
                 kernel_size,
                 pool_size,
                 dropout,
                 initializer_range,
                 scope=None):
  # (b, s, h, 1)
  inputs = tf.expand_dims(inputs, -1)

  # (b, s, h, 1)
  with tf.variable_scope('conv_1'):
    filter_shape = [kernel_size[0], kernel_size[0], 1, 1]
    h_1 = customized_cnn(inputs, filter_shape)
    # (b, s, 1, 1)
    pooled_max_1 = tf.nn.max_pool(h_1,
                                  ksize=[1, 1, pool_size[0], 1],
                                  strides=[1, 1, 1, 1],
                                  padding='VALID',
                                  name='max_pool')
    # (b, s, 1, 1)
    pooled_avg_1 = tf.nn.avg_pool(h_1,
                                ksize=[1, 1, pool_size[0], 1],
                                strides=[1, 1, 1, 1],
                                padding='VALID',
                                name='avg_pool')

  # (b, s-2+1, h-2+1, 1)
  with tf.variable_scope('conv_2'):
    filter_shape = [kernel_size[1], kernel_size[1], 1, 1]
    h_2 = customized_cnn(h_1, filter_shape)
    # (b, s-2+1, 1, 1)
    pooled_max_2 = tf.nn.max_pool(h_2,
                                  ksize=[1, 1, pool_size[1], 1],
                                  strides=[1, 1, 1, 1],
                                  padding='VALID',
                                  name='max_pool')
    # (b, s-2+1, 1, 1)
    pooled_avg_2 = tf.nn.avg_pool(h_2,
                                ksize=[1, 1, pool_size[1], 1],
                                strides=[1, 1, 1, 1],
                                padding='VALID',
                                name='avg_pool')
  
  # (b, s-3+1, h-3+1, 1)
  with tf.variable_scope('conv_3'):
    filter_shape = [kernel_size[2], kernel_size[2], 1, 1]
    h_3 = customized_cnn(h_2, filter_shape)
    # (b, s-3+1, 1, 1)
    pooled_max_3 = tf.nn.max_pool(h_3,
                                  ksize=[1, 1, pool_size[2], 1],
                                  strides=[1, 1, 1, 1],
                                  padding='VALID',
                                  name='max_pool')

    # (b, s-3+1, 1, 1)
    pooled_avg_3 = tf.nn.avg_pool(h_3,
                                ksize=[1, 1, pool_size[2], 1],
                                strides=[1, 1, 1, 1],
                                padding='VALID',
                                name='avg_pool')

  pooled_max_1 = tf.squeeze(pooled_max_1)
  pooled_avg_1 = tf.squeeze(pooled_avg_1)
  pooled_max_2 = tf.squeeze(pooled_max_2)
  pooled_avg_2 = tf.squeeze(pooled_avg_2)
  pooled_max_3 = tf.squeeze(pooled_max_3)
  pooled_avg_3 = tf.squeeze(pooled_avg_3)

  # (b, s-4)
  output = tf.concat((pooled_max_1, pooled_avg_1,
                      pooled_max_2, pooled_avg_2,
                      pooled_max_3, pooled_avg_3),
                      axis=-1)
  return output

def customized_cnn(inputs, filter_shape):
  with tf.variable_scope('conv_1'):
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
    b = tf.Variable(tf.truncated_normal([1], stddev=0.1), name='b')
    conv = tf.nn.conv2d(inputs,
                        W,
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='conv')
    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
    return h

def AttentionLayer(query,
                   value):
    attn_scores = tf.matmul(query, value, transpose_b=True)
    attn_probs = tf.nn.softmax(attn_scores, axis=-1)
    context = tf.matmul(attn_probs, value)

    return context


if __name__ == '__main__':
  class Config():
    vocab_size = 32
    embedding_size = 32
    num_layers = 4
    hidden_size = 32
    forget_bias = 1.0

    kernel_size = [1, 2, 3]
    pool_size = [64, 63, 61]

    initializer_range = 0.01
    dropout = 0.1
  
  config = Config()
  sent_A = tf.constant([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]], dtype=tf.int32)
  sent_B = tf.constant([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]], dtype=tf.int32)
  sent_length_A = tf.constant([6, 6], dtype=tf.int32)
  sent_length_B = tf.constant([6, 6], dtype=tf.int32)

  model = ERCNNModel(config, True, sent_A, sent_B, sent_length_A, sent_length_B)