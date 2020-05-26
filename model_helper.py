# coding:utf-8

import sys
import six
import tensorflow as tf

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

import config as _cg
from utils.log import log_info as _info
from utils.log import log_error as _error

"""TENSOR CALCULATE"""
def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of shape of tensor, preferring static dimensions,
        sometimes, the dimension is None.
             
    Args:
        tensor: a tf.Tensor which needs to find the shape.
        expected_rank: (optional) int. The expected rank of 'tensor'. If this is
            specified and the 'tensor' has a different rank, an error will be thrown.
        name: (optional) name of the 'tensor' when throwing the error.

    Returns:
        Dimensions as list type of the tensor shape.
        All static dimension will be returned as python integers,
        and dynamic dimensions will be returned as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)
    
    shape = tensor.shape.as_list()

    # save the dimension which is None
    dynamic_indices = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            dynamic_indices.append(index)
    
    # dynamic_indices list is None
    if not dynamic_indices:
        return shape
    
    # replace the dynamic dimensions
    dynamic_shape = tf.shape(tensor)
    for index in dynamic_indices:
        shape[index] = dynamic_shape[index]
    return shape

def assert_rank(tensor, expected_rank, name=None):
    """Check whether the rank of the 'tensor' matches the expecetd_rank.
        Remember rank is the number of the total dimensions.
        
    Args:
        tensor: A tf.tensor to check.
        expected_rank: Python integer or list of integers.
        name: (optional) name of the 'tensor' when throwing the error.    
    """
    if name is None:
        name = tensor.name
    
    expected_rank_dict = {}
    # save the given rank into the dictionary, 
    # given rank could be either an integer or a list.
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for rank in expected_rank:
            expected_rank_dict[rank] = True

    tensor_rank = tensor.shape.ndims
    if tensor_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        _error('For the tensor {} in scope {}, the tensor rank {%d} \
                (shape = {}) is not equal to the expected_rank {}'.format(
            name, scope_name, tensor_rank, str(tensor.shape), str(expected_rank)))
        raise ValueError

def create_initializer(initializer_range=0.02):
    """Initialzer Variable."""
    return tf.truncated_normal_initializer(stddev=initializer_range)

"""Model Relevant"""
def create_embedding(vocab_size, embedding_size, name, initializer_range):
    embedding_table = tf.get_variable(name=name,
                                      shape=[vocab_size, embedding_size],
                                      initializer=create_initializer(initializer_range))
    return embedding_table

def create_single_cell_RNN(unit_type,
                           num_units,
                           dropout,
                           residual,
                           forget_bias):
    """create single cell for rnn."""
    single_cell = None
    ac = tf.nn.tanh

    if unit_type is _cg.RNN_UNIT_TYPE_LSTM:
        single_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units, forget_bias=forget_bias, activation=ac)
    elif unit_type is _cg.RNN_UNIT_TYPE_GRU:
        single_cell = tf.contrib.rnn.GRUCell(num_units, activation=ac)
    elif unit_type is _cg.RNN_UNIT_TYPE_LAYER_NORM_LSTM:
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units, forget_bias=forget_bias, layer_norm=True, activation=ac)
    else:
        _error('Unit Type: {} not support.'.format(unit_type))
        raise ValueError

    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - dropout))

    if residual:
        single_cell = tf.contrib.rnn.ResidualWrapper(single_cell)
    
    return single_cell

def create_cell_list_for_RNN(unit_type, 
                             num_units, 
                             num_layers, 
                             dropout, 
                             num_residual_layers,
                             forget_bias):
    """Create cells for rnn.
    
    Args:
        unit_type: str type choice from ['lstm', 'gru', 'layer_norm_lstm'].
        num_units: int type of hidden size.
        num_layers: the number of layers of each rnn block.
        dropout: float type, set to 0 when not training.
        num_residual_layers: int.
        forget_bias: float.
    
    Returns:
        A single cell or a wrapper of cells.
    """
    cell_list = []

    for i in range(num_layers):
        cell = create_single_cell_RNN(unit_type, num_units, dropout,
                                      residual=(
                                         i >= num_layers - num_residual_layers),
                                      forget_bias=forget_bias)
        cell_list.append(cell)
    
    return cell_list[0] if len(cell_list) == 1 else tf.contrib.rnn.MultiRNNCell(cell_list)


"""VAE"""
def vae(state, num_units, scope='vae'):
    """VAE implementation, Hard Coding."""
    # states = [state[0], state[1], state[2], state[3]]
    states = [state[0], state[1]]
    states = tf.transpose(state, [1, 0, 2])
    shape = get_shape_list(states)

    with tf.variable_scope(scope):
        vae_mean = tf.layers.dense(states,
                                        num_units,
                                        activation=tf.nn.tanh,
                                        name='vae_mean',
                                        kernel_initializer=create_initializer())

        vae_vb = tf.layers.dense(states,
                                      num_units,
                                      activation=tf.nn.tanh,
                                      name='vae_vb',
                                      kernel_initializer=create_initializer())
        
        eps = tf.random_normal([shape[0], shape[1], num_units], 0.0, 1.0, dtype=tf.float32)

        z = vae_mean + tf.sqrt(tf.exp(vae_vb)) * eps
    
    states_list = []
    for i in range(_cg.nmt_config.num_layers):
        states_list.append(z[:, i, :])
    return tuple(states_list), vae_mean, vae_vb