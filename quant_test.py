# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
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
#
# Modifications Copyright 2017-2018 Arm Inc. All Rights Reserved. 
# Adapted from freeze.py to run quantized inference on train/val/test dataset on the 
# trained model in the form of checkpoint
#          
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import numpy as np

import tensorflow as tf
import input_data
import quant_models as models

def run_quant_inference(wanted_words, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms, dct_coefficient_count, 
                           model_architecture, model_size_info):
  """Creates an audio model with the nodes needed for inference.

  Uses the supplied arguments to create a model, and inserts the input and
  output nodes that are needed to use the graph for inference.

  Args:
    wanted_words: Comma-separated list of the words we're trying to recognize.
    sample_rate: How many samples per second are in the input audio files.
    clip_duration_ms: How many samples to analyze for the audio pattern.
    window_size_ms: Time slice duration to estimate frequencies from.
    window_stride_ms: How far apart time slices should be.
    dct_coefficient_count: Number of frequency bands to analyze.
    model_architecture: Name of the kind of model to generate.
    model_size_info: Model dimensions : different lengths for different models
  """
  
  tf.logging.set_verbosity(tf.logging.INFO)
  sess = tf.InteractiveSession()
  words_list = input_data.prepare_words_list(wanted_words.split(','))
  model_settings = models.prepare_model_settings(
      len(words_list), sample_rate, clip_duration_ms, window_size_ms,
      window_stride_ms, dct_coefficient_count)

  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
      FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings)
  
  label_count = model_settings['label_count']
  fingerprint_size = model_settings['fingerprint_size']
  time_shift_samples = int((100.0 * FLAGS.sample_rate) / 1000)

  fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')

  logits = models.create_model(
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      FLAGS.model_size_info,
      FLAGS.act_max,
      is_training=False)
  ground_truth_input = tf.placeholder(
    tf.float32, [None, label_count], name='groundtruth_input')    

  if FLAGS.if_retrain:
    with tf.name_scope('cross_entropy'):
      cross_entropy_mean = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input, logits=logits))
    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.name_scope('train'), tf.control_dependencies(update_ops):
      train_op = tf.train.AdamOptimizer(learning_rate=0.0001)
      train_step = tf.contrib.slim.learning.create_train_op(cross_entropy_mean, train_op)

  saver = tf.train.Saver(tf.global_variables())
  merged = tf.summary.merge_all()
  test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test', sess.graph)
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train')
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')
  tf.global_variables_initializer().run()

  predicted_indices = tf.argmax(logits, 1)
  expected_indices = tf.argmax(ground_truth_input, 1)
  correct_prediction = tf.equal(predicted_indices, expected_indices)
  confusion_matrix = tf.confusion_matrix(
      expected_indices, predicted_indices, num_classes=label_count)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  models.load_variables_from_checkpoint(sess, FLAGS.checkpoint)

  for v in tf.trainable_variables():
    var_name = str(v.name)
    var_values = sess.run(v)
    min_value = var_values.min()
    max_value = var_values.max()
    int_bits = int(np.ceil(np.log2(max(abs(min_value),abs(max_value)))))
    # ap_fixed<8,1> uses 7 decimal bits and 1 bit for sign
    dec_bits = 7-int_bits
    # dec_bits = min(7, 7-int_bits)
    # convert to [-128,128) or int8
    # var_values = np.round(var_values*2**dec_bits)
    # convert back original range but quantized to 8-bits or 256 levels
    # var_values = var_values/(2**dec_bits)
    if FLAGS.update_weights:
      # define datatypes
      # f = open('weights/parameters.h','wb')
      # f.close()
      from save_data import prepare_header, prepare_lstm_headers
      var_name_split = var_name.split(':')
      if var_name_split[0].startswith('W_o'):
        os.makedirs('weights/fc', exist_ok=True)
        c_var_name = 'Wy['+str(var_values.shape[1]) + '][' +str(var_values.shape[0]) + ']' # transposed
        np.savetxt('weights/fc/Wy.h', np.transpose(var_values), delimiter=',', newline=',\n')
        prepare_header('weights/fc/Wy.h', 'Wy_t '+c_var_name)
      elif var_name_split[0].startswith('b_o'):
        c_var_name = 'by['+str(var_values.shape[0]) + ']'
        np.savetxt('weights/fc/by.h', var_values[None], delimiter=',')
        prepare_header('weights/fc/by.h','by_t '+c_var_name)

      elif var_name_split[0].startswith('lstm'):
        lstm_name = var_name_split[0].split('/')
        param_name = lstm_name[-1]
        # if (lstm_name[0] == 'lstm0'):
        #   prepare_lstm_headers('weights/' + lstm_name[0], var_values,input_size = FLAGS.dct_coefficient_count, param_name=param_name)
        # else: 
        #   state_size = FLAGS.model_size_info[0] # TODO
        #   prepare_lstm_headers('weights/' + lstm_name[0], var_values,input_size = state_size, param_name=param_name) 

        # for lstmp
        if (lstm_name[-2] == 'projection'):
          param_name = 'projection'
        if (lstm_name[1] == 'lstm0'):
          prepare_lstm_headers('weights/' + lstm_name[1], var_values,input_size = FLAGS.dct_coefficient_count, param_name=param_name)
        else: 
          state_size = FLAGS.model_size_info[0] # TODO
          prepare_lstm_headers('weights/' + lstm_name[1], var_values,input_size = state_size, param_name=param_name) 

    # update the weights in tensorflow graph for quantizing the activations
    var_values = sess.run(tf.assign(v,var_values))    
    print(var_name+' number of wts/bias: '+str(var_values.shape)+\
            ' dec bits: '+str(dec_bits)+\
            ' max: ('+str(var_values.max())+','+str(max_value)+')'+\
            ' min: ('+str(var_values.min())+','+str(min_value)+')')
  if FLAGS.if_retrain:
    best_accuracy = 0
    for training_step in range(FLAGS.retrain_steps):
      # Pull the audio samples we'll use for training.
      train_fingerprints, train_ground_truth = audio_processor.get_data(
          FLAGS.batch_size, 0, model_settings, 0.8,
          0.1, time_shift_samples, 'training', sess)
      # Run the graph with this batch of training data.
      train_summary, train_accuracy, cross_entropy_value, _ = sess.run(
          [
              merged, evaluation_step, cross_entropy_mean, train_step
          ],
          feed_dict={
              fingerprint_input: train_fingerprints,
              ground_truth_input: train_ground_truth
          })
      train_writer.add_summary(train_summary, training_step)
      tf.logging.info('Step #%d: accuracy %.2f%%, cross entropy %f' %
                      (training_step, train_accuracy * 100,
                      cross_entropy_value))
      is_last_step = (training_step == FLAGS.retrain_steps)
      if (training_step % 200) == 0 or is_last_step:
        set_size = audio_processor.set_size('validation')
        total_accuracy = 0
        total_conf_matrix = None
        for i in range(0, set_size, FLAGS.batch_size):
          validation_fingerprints, validation_ground_truth = (
              audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
                                      0.0, 0, 'validation', sess))

          # Run a validation step and capture training summaries for TensorBoard
          # with the `merged` op.
          validation_summary, validation_accuracy, conf_matrix = sess.run(
              [merged, evaluation_step, confusion_matrix],
              feed_dict={
                  fingerprint_input: validation_fingerprints,
                  ground_truth_input: validation_ground_truth
              })
          validation_writer.add_summary(validation_summary, training_step)
          batch_size = min(FLAGS.batch_size, set_size - i)
          total_accuracy += (validation_accuracy * batch_size) / set_size
          if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
          else:
            total_conf_matrix += conf_matrix
        tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
        tf.logging.info('Step %d: Validation accuracy = %.2f%% (N=%d)' %
                        (training_step, total_accuracy * 100, set_size))

        # Save the model checkpoint when validation accuracy improves
        if total_accuracy > best_accuracy:
          best_accuracy = total_accuracy
          checkpoint_path = os.path.join(FLAGS.new_checkpoint,
                                        FLAGS.model_architecture + '_'+ str(int(best_accuracy*10000)) + '.ckpt')
          tf.logging.info('Saving best model to "%s-%d"', checkpoint_path, training_step)
          saver.save(sess, checkpoint_path, global_step=training_step)
        tf.logging.info('So far the best validation accuracy is %.2f%%' % (best_accuracy*100))

  # validation set
  set_size = audio_processor.set_size('validation')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in range(0, set_size, FLAGS.batch_size):
    validation_fingerprints, validation_ground_truth = (
        audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
                                 0.0, 0, 'validation', sess))
    validation_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            fingerprint_input: validation_fingerprints,
            ground_truth_input: validation_ground_truth,
        })
    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (validation_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Validation accuracy = %.2f%% (N=%d)' %
                  (total_accuracy * 100, set_size))
  
  # test set
  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in range(0, set_size, FLAGS.batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
    
    test_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            fingerprint_input: test_fingerprints,
            ground_truth_input: test_ground_truth,
        })
    
    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (test_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix

  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Test accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
                                                           set_size))

def main(_):

  # Create the model, load weights from checkpoint and run on train/val/test
  run_quant_inference(FLAGS.wanted_words, FLAGS.sample_rate,
      FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count,
      FLAGS.model_architecture, FLAGS.model_size_info)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='./speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='./quant/summary',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=16,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      # GIVE PATH LIKE ./train/ckpts/best/basic_lstm_9358.ckpt-16000
      '--checkpoint',
      type=str,
      default='',
      help='Checkpoint to load the weights from.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='basic_lstm',
      help='What model architecture to use')
  parser.add_argument(
      '--model_size_info',
      type=int,
      nargs="+",
      default=[128],
      help='Model dimensions - different for various models')
  parser.add_argument(
      '--act_max',
      type=float,
      nargs="+",
      default=[32,32],
      help='activations max')
  parser.add_argument(
      '--if_retrain',
      type=int,
      default=0)
  parser.add_argument(
      '--retrain_steps',
      type=int,
      default=3000)    
  parser.add_argument(
      '--new_checkpoint',
      type=str,
      default='./quant/ckpts',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument('--update_weights', type=int, default=0)       

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
