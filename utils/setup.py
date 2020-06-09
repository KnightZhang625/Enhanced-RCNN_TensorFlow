# coding:utf-8

import sys
import logging
import tensorflow as tf
from pathlib import Path

PROJECT_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_PATH))

class Setup(object):
    """Setup logging"""
    def __init__(self, log_name='tensorflow', path=str(PROJECT_PATH / 'log')):
        Path(PROJECT_PATH / 'log').mkdir(exist_ok=True)
        tf.compat.v1.logging.set_verbosity(logging.INFO)
        handlers = [logging.FileHandler(str(PROJECT_PATH / 'log/main.log')),
                    logging.StreamHandler(sys.stdout)]
        logging.getLogger('tensorflow').handlers = handlers