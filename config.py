# coding:utf-8

import sys
from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from utils.log import log_info as _info
from utils.log import log_error as _error

RNN_ENCODER_TYPE_UNI = 'uni'
RNN_ENCODER_TYPE_BI = 'bi'

RNN_UNIT_TYPE_LSTM = 'lstm'
RNN_UNIT_TYPE_GRU = 'gru'
RNN_UNIT_TYPE_LAYER_NORM_LSTM = 'layer_norm_lstm'

SOS_ID = 1
EOS_ID = 2
PADDING_ID = 3

BATCH_SIZE = 64
TRIAN_STEPS = 100000
QUE_PATH = PROJECT_PATH / 'data/question.data'
ANS_PATH = PROJECT_PATH / 'data/answer.data'

def forbid_new_attributes(wrapped_setatrr):
    def __setattr__(self, name, value):
        if hasattr(self, name):
            wrapped_setatrr(self, name, value)
        else:
            _error('Add new {} is forbidden'.format(name))
            raise AttributeError
    return __setattr__

class NoNewAttrs(object):
    """forbid to add new attributes"""
    __setattr__ = forbid_new_attributes(object.__setattr__)
    class __metaclass__(type):
        __setattr__ = forbid_new_attributes(type.__setattr__)

class NmtConfig(NoNewAttrs):
        # Encoder
        src_vocab_size = 7819
        embedding_size = 32
        num_layers = 2
        num_units = 32
        forget_bias = 1.0
        dropout = 0.2
        residual_or_not = True  

        # Decoder
        tgt_vocab_size = 7819 
        max_len_infer = 50

        encoder_type = RNN_ENCODER_TYPE_BI
        unit_type = RNN_UNIT_TYPE_GRU

        # global
        model_dir = 'models/'
        initializer_range = 0.02
        learning_rate = 5e-3
        lr_limit = 5e-3
        colocate_gradients_with_ops = True

        segment_size = 2

nmt_config = NmtConfig()