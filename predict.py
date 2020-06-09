# coding:utf-8

import sys
import codecs
import pickle
import functools
from pathlib import Path
from tensorflow.contrib import predictor

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from load_data import create_batch_idx, convert_str_idx, padding_func

def restore_model(pb_path):
  """Restore the latest model from the given path."""
  subdirs = [x for x in Path(pb_path).iterdir()
             if x.is_dir() and 'temp' not in str(x)]
  latest_model = str(sorted(subdirs)[-1])
  predict_fn = predictor.from_saved_model(latest_model)

  return predict_fn

def predict(model, data_path, batch_size):
  with codecs.open(data_path, 'rb') as file:
    data = pickle.load(file)
  
  for start, end in create_batch_idx(len(data), batch_size):
    input_batch = data[start: end]
    input_a = list(map(convert_str_idx, input_batch))
    input_length = list(map(len, input_a))
    max_length = max(input_length)
    padding_func_with_args = functools.partial(padding_func, max_length=max_length)
    input_padded = list(map(padding_func_with_args, input_a))

    features = {'input_A': input_padded,
                'input_A_length': input_length}
    
    predictions = model(features)
    output_vectors = predictions['output_vector']
    print(output_vectors)
    input()

if __name__ == '__main__':
  model = restore_model(PROJECT_PATH / 'models_deployed')
  predict(model, PROJECT_PATH / 'data/retrieve_data_60K.bin', 32)
