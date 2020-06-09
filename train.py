# coding:utf-8

import sys
import functools
import numpy as np
import tensorflow as tf

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from utils.setup import Setup
from utils.log import log_info as _info
from utils.log import log_error as _error
setup = Setup()

import config as _cg
import model_helper as _mh
from Model import ERCNNModel
from config import nmt_config
from load_data import train_input_fn
from load_data import server_input_receiver_fn
# from optimization import create_optimizer
# from data_pipeline_for_nmt import train_input_fn, server_input_receiver_fn

# def get_ppl(logtis):
#     prob = logtis   # [b, s, h]
#     prob = tf.nn.softmax(prob, axis=-1)
#     ids = tf.argmax(prob, axis=-1)  # [b, s]
#     one_hot_ids = tf.one_hot(ids, _mh.get_shape_list(prob)[-1], dtype=tf.float32)    # [b, s, h]
#     prob = prob * one_hot_ids
#     sprob = tf.reduce_sum(prob, axis=-1)
#     ppl = tf.reduce_mean(-tf.log(sprob) * 100.)
#     return ppl

def cosine_similarity(vector_a, vector_b):
	denominator_a = tf.sqrt(tf.reduce_sum(tf.multiply(vector_a, vector_a), axis=-1))
	denominator_b = tf.sqrt(tf.reduce_sum(tf.multiply(vector_b, vector_b), axis=-1))
	numerator = tf.reduce_sum(tf.multiply(vector_a, vector_b), axis=-1)
	cosine = tf.div(numerator, denominator_a * denominator_b + 1e-8, name='cosine')
	return cosine


def sigmoid(x):
	return 1.0 / (1 + tf.exp(-x))


def model_fn_builder(config):
		"""Returns 'model_fn' closure for Estimator."""

		def model_fn(features, labels, mode, params):
				# obtain the data
				_info('*** Features ***')
				for name in sorted(features.keys()):
						tf.logging.info(' name = %s, shape = %s' % (name, features[name].shape))
				
				is_training = (mode == tf.estimator.ModeKeys.TRAIN)

				if is_training:
					input_A = features['input_A']
					input_B = features['input_B']
					input_A_length = features['input_A_length']
					input_B_length = features['input_B_length']
				else:
					input_A = features['input_A']
					input_B = features['input_A']
					input_A_length = features['input_A_length']
					input_B_length = features['input_A_length']
					

				# if mode != tf.estimator.ModeKeys.PREDICT:
				#     decoder_input_data = features['decoder_input_data']
				#     seq_length_decoder_input_data = features['seq_length_decoder_input_data']
				# else:
				#     decoder_input_data = None
				#     seq_length_decoder_input_data = None

				# build Encoder
				model = ERCNNModel(config=config,
													 is_training=is_training,
													 sent_A=input_A,
													 sent_B=input_B,
													 sent_length_A=input_A_length,
													 sent_length_B=input_B_length)

				output = model.get_output()

				# [b, s]
				batch_size = tf.cast(_mh.get_shape_list(output)[0], dtype=tf.float32)
				# output = tf.reduce_sum(tf.multiply(output_A, output_B), axis=-1)
				# output = tf.reshape(output, (batch_size, 1))

				if mode == tf.estimator.ModeKeys.PREDICT:
						predictions = {'output_vector': output}
						# the default key in 'output', however, when customized, the keys are identical with the keys in dict.
						output_spec = tf.estimator.EstimatorSpec(mode, predictions=predictions)
				else:
						if mode == tf.estimator.ModeKeys.TRAIN:
								# labels = tf.cast(labels, tf.float32)
								# loss = tf.losses.mean_squared_error(labels, output)

								# loss = tf.losses.mean_squared_error(output_A, output_B)

								loss = tf.reduce_sum(
								        tf.nn.sparse_softmax_cross_entropy_with_logits(
								            labels=labels, logits=output)) / batch_size 
								# # loss = vae_loss + seq_loss
								# loss = seq_loss
								
								"""
								Tutorial on `polynomial_decay`:
										The formula is as below:
													
													global_step = min(global_step, decay_steps)
													decayed_learning_rate = (learning_rate - end_learning_rate) * (1 - global_step / decay_steps) ^ (power) + end_learning_rate
										
										global_step: each batch step.
										decay_steps: the whole step, the lr will touch the end_learning_rate after the decay_steps.
										TRAIN_STEPS: the number for repeating the whole dataset, so the decay_steps = len(dataset) / batch_size * TRAIN_STEPS.
								"""
								# train_op, lr = create_optimizer(loss, config.learning_rate, _cg.TRIAN_STEPS, config.lr_limit)
								
								
								learning_rate = tf.train.polynomial_decay(config.learning_rate,
																													tf.train.get_or_create_global_step(),
																													_cg.TRIAN_STEPS,
																													end_learning_rate=0.0,
																													power=1.0,
																													cycle=False)

								lr = tf.maximum(tf.constant(config.lr_limit), learning_rate)
								optimizer = tf.train.AdamOptimizer(lr, name='optimizer')
								tvars = tf.trainable_variables()
								gradients = tf.gradients(loss, tvars, colocate_gradients_with_ops=config.colocate_gradients_with_ops)
								clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
								train_op = optimizer.apply_gradients(zip(clipped_gradients, tvars), global_step=tf.train.get_global_step())
								

								# this is excellent, because it could display the result each step, i.e., each step equals to batch_size.
								# the output_spec, display the result every save checkpoints step.
								logging_hook = tf.train.LoggingTensorHook({'loss' : loss, 'lr': lr}, every_n_iter=10)

								output_spec = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])

						elif mode == tf.estimator.ModeKeys.EVAL:
								# TODO
								raise NotImplementedError
				
				return output_spec
		
		return model_fn

def main():
		Path(nmt_config.model_dir).mkdir(exist_ok=True)

		model_fn = model_fn_builder(nmt_config)
		
		gpu_config = tf.ConfigProto()
		gpu_config.gpu_options.allow_growth = True
		
		# run_config = tf.contrib.tpu.RunConfig(
		#     keep_checkpoint_max=1,
		#     save_checkpoints_steps=1000,
		#     model_dir=nmt_config.model_dir)

		run_config = tf.estimator.RunConfig(		
			session_config=gpu_config,
			keep_checkpoint_max=1,
			save_checkpoints_steps=1000,
			model_dir=nmt_config.model_dir)
		
		estimaotr = tf.estimator.Estimator(model_fn, config=run_config)
		estimaotr.train(train_input_fn)     # train_input_fn should be callable

def package_model(ckpt_path, pb_path):
		model_fn = model_fn_builder(nmt_config)
		estimator = tf.estimator.Estimator(model_fn, ckpt_path)
		estimator.export_saved_model(pb_path, server_input_receiver_fn)

if __name__ == '__main__':
		if sys.argv[1] == 'train':
				main()
		elif sys.argv[1] == 'package':
				package_model(str(PROJECT_PATH / 'models'), str(PROJECT_PATH / 'models_deployed'))
		else:
				_error('Unknown parameter: {}.'.format(sys.argv[1]))
				_info('Choose from [train | package].')