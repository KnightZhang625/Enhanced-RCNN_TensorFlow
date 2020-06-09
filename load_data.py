# coding:utf-8

import sys
import copy
import pickle
import codecs
import random
import functools
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

import config as _cg

# vocab_idx = {}
# with codecs.open('data/smvocab.data', 'r', 'utf-8') as file:
#   for i, line in enumerate(file):
#     line = line.strip()
#     vocab_idx[line] = i
# print(vocab_idx)
# with codecs.open('data/vocab_idx.bin', 'wb') as file:
#   pickle.dump(vocab_idx, file)

# data_all = []
# with codecs.open('data/all.data', 'r', 'utf-8') as file:
#   for line in file:
#     line = line.strip().split('=')
#     que, ans = line[0], line[1]
#     data_all.append((que, ans))
# print(len(data_all))
# with codecs.open(_cg.DATA_PATH, 'wb') as file:
#   pickle.dump(data_all, file)

def load_dict():
	global vocab_idx
	with codecs.open(_cg.DIST_PATH, 'rb') as file:
		vocab_idx = pickle.load(file)
load_dict()

def create_batch_idx(data_length, batch_size):
	batch_number = data_length // batch_size
	batch_number = batch_number if data_length % batch_size == 0 else batch_number + 1

	for i in range(batch_number):
		yield (i * batch_size, i * batch_size + batch_size)

convert_str_idx = lambda string : [vocab_idx[v] if v in vocab_idx else vocab_idx['<unk>'] for v in string]
padding_func = lambda data, max_length=30: data + [vocab_idx['<padding>'] \
																									for _ in range(max_length - len(data))]
remove_function = lambda que, ans, f_ans : [(q, a, f_a) for q, a, f_a in zip(que, ans, f_ans) \
	 if len(q) <= 30 and len(a) <= 30 and len(f_a) <= 30]

def train_generator():
	with codecs.open(_cg.DATA_PATH, 'rb') as file:
		data = pickle.load(file)
	random.shuffle(data)
	
	for start, end in create_batch_idx(len(data), _cg.BATCH_SIZE):
		data_batch = data[start:end]

		que = [item[0] for item in data_batch]
		ans = [item[1] for item in data_batch]

		que_idx = list(map(convert_str_idx, que))
		ans_idx = list(map(convert_str_idx, ans))
		fake_ans_idx = list(reversed(ans_idx))
		mid_idx = len(data_batch) // 2
		fake_ans_idx[mid_idx], fake_ans_idx[-1] = fake_ans_idx[-1], fake_ans_idx[mid_idx]

		que_ans_fakeans = remove_function(que_idx, ans_idx, fake_ans_idx)
		que_idx, ans_idx, fake_ans_idx = zip(*que_ans_fakeans)

		que_idx_padded = list(map(padding_func, que_idx))
		ans_idx_padded = list(map(padding_func, ans_idx))
		fake_ans_idx_padded = list(map(padding_func, fake_ans_idx))

		input_A = que_idx_padded + que_idx_padded
		input_A_length = [len(que) for que in input_A]
		input_B = ans_idx_padded + fake_ans_idx_padded
		input_B_length = [len(ans) for ans in input_B]
		labels = [1 for _ in range(len(que_idx_padded))] + [0 for _ in range(len(ans_idx_padded))]

		# que_length = list(map(len, que_idx))
		# max_length_que = max(que_length)
		# padding_que = functools.partial(padding_func, max_length= max_length_que)
		# ans_length = list(map(len, ans_idx))
		# fake_ans_length = list(map(len, fake_ans_idx))
		# max_length_ans = max(ans_length)
		# padding_ans = functools.partial(padding_func, max_length= max_length_ans)

		# que_idx_padded = list(map(padding_que, que_idx))
		# ans_idx_padded = list(map(padding_ans, ans_idx))
		# fake_ans_idx_padded = list(map(padding_ans, fake_ans_idx))

		# input_A = que_idx_padded + que_idx_padded
		# input_A_length = que_length + que_length
		# input_B = ans_idx_padded + fake_ans_idx_padded
		# input_B_length = ans_length + fake_ans_length
		# labels = [1 for _ in range(len(que_idx_padded))] + [0 for _ in range(len(ans_idx_padded))]

		pairs = list(zip(input_A, input_A_length, input_B, input_B_length, labels))
		random.shuffle(pairs)
		input_A, input_A_length, input_B, input_B_length, labels = zip(*pairs)

		features = {'input_A': np.array(input_A, dtype=np.int32),
								'input_B': np.array(input_B, dtype=np.int32),
								'input_A_length': np.array(input_A_length, dtype=np.int32),
								'input_B_length': np.array(input_B_length, dtype=np.int32)}
		
		yield (features, np.array(labels, dtype=np.int32))

def train_input_fn():
	output_types = {'input_A': tf.int32,
									'input_B': tf.int32,
									'input_A_length': tf.int32,
									'input_B_length': tf.int32}
	
	output_shapes = {'input_A': [None, None],
									 'input_B': [None, None],
									 'input_A_length': [None],
									 'input_B_length': [None]}
	
	dataset = tf.data.Dataset.from_generator(
		train_generator,
		output_types=(output_types, tf.int32),
		output_shapes=(output_shapes, [None]))

	dataset = dataset.repeat(_cg.TRIAN_STEPS)

	return dataset

def server_input_receiver_fn():
	input_A = tf.placeholder(tf.int32, shape=[None, None], name='input_A')
	input_A_length = tf.placeholder(tf.int32, shape=[None], name='input_A_length')

	receiver_tensors = {'input_A': input_A,
											'input_A_length': input_A_length}
	features = {'input_A': input_A,
							'input_A_length': input_A_length}
	
	return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

if __name__ == '__main__':
	for data in train_input_fn():
		print(data)
		input()