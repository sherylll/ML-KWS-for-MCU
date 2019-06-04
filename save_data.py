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
# Modifications Copyright 2017 Arm Inc. All Rights Reserved. 
# Adapted from freeze.py to run inference on train/val/test dataset on the 
# trained model in the form of checkpoint
#          
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import tensorflow as tf
import input_data
import models

import numpy as np
import os
import matplotlib.pyplot as plt


def prepare_header(filename,definition):
  data = definition + '={'
  with open(filename) as f:
      data = data + f.read()+'};'
  with open(filename, 'w+') as f:
          f.write(data)
def prepare_lstm_headers(folder, var_values, input_size = None, param_name=None):
  os.makedirs(folder, exist_ok=True)
  if param_name == 'bias': # bias (i, j, f, o)
    bias_size = int(var_values.shape[0]/4)
    np.savetxt(folder + '/bi.h', var_values[:bias_size][None], delimiter=',')
    np.savetxt(folder + '/bc.h', var_values[bias_size:2*bias_size][None], delimiter=',')
    np.savetxt(folder + '/bf.h', var_values[2*bias_size:3*bias_size][None], delimiter=',')
    np.savetxt(folder + '/bo.h', var_values[3*bias_size:][None], delimiter=',')
    prepare_header(folder + '/bi.h','bias_t bi['+str(bias_size)+']')
    prepare_header(folder + '/bc.h','bias_t bc['+str(bias_size)+']')
    prepare_header(folder + '/bf.h','bias_t bf['+str(bias_size)+']')
    prepare_header(folder + '/bo.h','bias_t bo['+str(bias_size)+']')
  elif param_name == 'projection': # kernel  
    np.savetxt(folder + '/proj.h', np.transpose(var_values), delimiter=',',newline=',\n')
    prepare_header(folder + '/proj.h','proj_t proj['+str(var_values.shape[1])+']'+'['+str(var_values.shape[0])+']')
  elif param_name == 'kernel': # kernel  
    hidden_size = int(var_values.shape[1]/4)
    if not input_size:
      input_size = hidden_size
    W_values = var_values[:input_size] # (mfcc,516)
    U_values = var_values[input_size:] # (128, 516)
    second_input_size = var_values.shape[0] - input_size
    W_size = '[' + str(hidden_size) + '][' + str(input_size) + ']'
    U_size = '[' + str(hidden_size) + '][' + str(second_input_size) + ']'
    W_transpose = np.transpose(W_values)
    U_transpose = np.transpose(U_values)
    # slicing
    Wi = W_transpose[:hidden_size]
    Wc = W_transpose[hidden_size:2*hidden_size]
    Wf = W_transpose[hidden_size*2:hidden_size*3]
    Wo = W_transpose[hidden_size*3:]
    Ui = U_transpose[:hidden_size]
    Uc = U_transpose[hidden_size:2*hidden_size]
    Uf = U_transpose[hidden_size*2:hidden_size*3]
    Uo = U_transpose[hidden_size*3:]
    # save to txt
    np.savetxt(folder + '/Wi.h', Wi, delimiter=',',newline=',\n')
    np.savetxt(folder + '/Wc.h', Wc, delimiter=',',newline=',\n')
    np.savetxt(folder + '/Wf.h', Wf, delimiter=',',newline=',\n')
    np.savetxt(folder + '/Wo.h', Wo, delimiter=',',newline=',\n')
    np.savetxt(folder + '/Ui.h', Ui, delimiter=',',newline=',\n')
    np.savetxt(folder + '/Uc.h', Uc, delimiter=',',newline=',\n')
    np.savetxt(folder + '/Uf.h', Uf, delimiter=',',newline=',\n')
    np.savetxt(folder + '/Uo.h', Uo, delimiter=',',newline=',\n')
    prepare_header(folder + '/Wi.h','kernel_t Wi'+W_size)
    prepare_header(folder + '/Wc.h','kernel_t Wc'+W_size)
    prepare_header(folder + '/Wf.h','kernel_t Wf'+W_size)
    prepare_header(folder + '/Wo.h','kernel_t Wo'+W_size)
    prepare_header(folder + '/Ui.h','kernel_t Ui'+U_size)
    prepare_header(folder + '/Uc.h','kernel_t Uc'+U_size)
    prepare_header(folder + '/Uf.h','kernel_t Uf'+U_size)
    prepare_header(folder + '/Uo.h','kernel_t Uo'+U_size)

def run_inference(wanted_words, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms, dct_coefficient_count, 
                           model_architecture, model_size_info):
  
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

  # test set
  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)

  test_fingerprints, test_ground_truth = audio_processor.get_data(
      set_size, 0, model_settings, 0.0, 0.0, 0, 'testing', sess, debugging=True, wav_path="speech_dataset\\up\\0a2b400e_nohash_0.wav")
  #for ii in range(set_size):
  #  np.savetxt('test_data/'+str(ii)+'.txt',test_fingerprints[ii], newline=' ', header=str(np.argmax(test_ground_truth[ii])))

  print(test_fingerprints)
#   reshaped_fingerprints = test_fingerprints.reshape((98,16)).transpose()
#   x = np.linspace(0, 98, 98)
#   for i in range(16):
    #   plt.plot(x, reshaped_fingerprints[i])
#   plt.xlabel('time step')
#   plt.ylabel('MFCC coeff.')
#   plt.show()
def main(_):

  # Create the model, load weights from checkpoint and run on train/val/test
  run_inference(FLAGS.wanted_words, FLAGS.sample_rate,
      FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count,
      FLAGS.model_architecture, FLAGS.model_size_info)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
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
      default='speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
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

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
