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
# Added new model definitions for speech command recognition used in
# the paper: https://arxiv.org/pdf/1711.07128.pdf
# Added Quantized model definitions 
#

"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import array_ops, clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import dtypes, ops, constant_op

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
  }


def create_model(fingerprint_input, model_settings, model_architecture,
                 model_size_info, act_max, is_training, \
                 runtime_settings=None):
  """Builds a model of the requested architecture compatible with the settings.

  There are many possible ways of deriving predictions from a spectrogram
  input, so this function provides an abstract interface for creating different
  kinds of models in a black-box way. You need to pass in a TensorFlow node as
  the 'fingerprint' input, and this should output a batch of 1D features that
  describe the audio. Typically this will be derived from a spectrogram that's
  been run through an MFCC, but in theory it can be any feature vector of the
  size specified in model_settings['fingerprint_size'].

  The function will build the graph it needs in the current TensorFlow graph,
  and return the tensorflow output that will contain the 'logits' input to the
  softmax prediction process. If training flag is on, it will also return a
  placeholder node that can be used to control the dropout amount.

  See the implementations below for the possible model architectures that can be
  requested.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    model_architecture: String specifying which kind of model to create.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
    Exception: If the architecture type isn't recognized.
  """
  if model_architecture == 'dnn':
    return create_dnn_model(fingerprint_input, model_settings, model_size_info,
                              act_max, is_training)
  elif model_architecture == 'basic_lstm':
    return create_basic_lstm_model(fingerprint_input, model_settings, model_size_info,
                              act_max, is_training) 
  elif model_architecture == 'lstm':
    return create_lstm_model(fingerprint_input, model_settings, model_size_info,
                              act_max, is_training)                                                        
  elif model_architecture == 'ds_cnn':
    return create_ds_cnn_model(fingerprint_input, model_settings, 
                                 model_size_info, act_max, is_training)
  elif model_architecture == 'crnn':
    return create_crnn_model(fingerprint_input, model_settings, model_size_info, act_max, is_training)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "single_fc", "conv",' +
                    ' "low_latency_conv", "low_latency_svdf",'+ 
                    ' "dnn", "cnn", "basic_lstm", "lstm",'+
                    ' "gru", "crnn" or "ds_cnn"')


def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)


def create_dnn_model(fingerprint_input, model_settings, model_size_info, 
                       act_max, is_training):
  """Builds a model with multiple hidden fully-connected layers.
  model_size_info: length of the array defines the number of hidden-layers and
                   each element in the array represent the number of neurons 
                   in that layer 
  """

  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  num_layers = len(model_size_info)
  layer_dim = [fingerprint_size]
  layer_dim.extend(model_size_info)
  flow = fingerprint_input
  if(act_max[0]!=0):
    flow = tf.fake_quant_with_min_max_vars(flow, min=-act_max[0], \
               max=act_max[0]-(act_max[0]/128.0), num_bits=8)
  for i in range(1, num_layers + 1):
      with tf.variable_scope('fc'+str(i)):
          W = tf.get_variable('W', shape=[layer_dim[i-1], layer_dim[i]], 
                initializer=tf.contrib.layers.xavier_initializer())
          b = tf.get_variable('b', shape=[layer_dim[i]])
          flow = tf.matmul(flow, W) + b
          if(act_max[i]!=0):
            flow = tf.fake_quant_with_min_max_vars(flow, min=-act_max[i], \
                       max=act_max[i]-(act_max[i]/128.0), num_bits=8)
          flow = tf.nn.relu(flow)
          if is_training:
            flow = tf.nn.dropout(flow, dropout_prob)

  weights = tf.get_variable('final_fc', shape=[layer_dim[-1], label_count], 
              initializer=tf.contrib.layers.xavier_initializer())
  bias = tf.Variable(tf.zeros([label_count]))
  logits = tf.matmul(flow, weights) + bias
  if(act_max[num_layers+1]!=0):
    logits = tf.fake_quant_with_min_max_vars(logits, min=-act_max[num_layers+1], \
                 max=act_max[num_layers+1]-(act_max[num_layers+1]/128.0), num_bits=8)
  if is_training:
    return logits, dropout_prob
  else:
    return logits

def create_ds_cnn_model(fingerprint_input, model_settings, model_size_info,
                          act_max, is_training):
  """Builds a model with depthwise separable convolutional neural network
  Model definition is based on https://arxiv.org/abs/1704.04861 and
  Tensorflow implementation: https://github.com/Zehaos/MobileNet

  model_size_info: defines number of layers, followed by the DS-Conv layer
    parameters in the order {number of conv features, conv filter height, 
    width and stride in y,x dir.} for each of the layers. 
  Note that first layer is always regular convolution, but the remaining 
    layers are all depthwise separable convolutions.
  """

  def ds_cnn_arg_scope(weight_decay=0):
    """Defines the default ds_cnn argument scope.
    Args:
      weight_decay: The weight decay to use for regularizing the model.
    Returns:
      An `arg_scope` to use for the DS-CNN model.
    """
    with slim.arg_scope(
        [slim.convolution2d, slim.separable_convolution2d],
        weights_initializer=slim.initializers.xavier_initializer(),
        biases_initializer=slim.init_ops.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
      return sc

  def _depthwise_separable_conv(inputs,
                                num_pwc_filters,
                                sc,
                                kernel_size,
                                stride,
                                layer_no,
                                act_max):
    """ Helper function to build the depth-wise separable convolution layer.
    """

    # skip pointwise by setting num_outputs=None
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=stride,
                                                  depth_multiplier=1,
                                                  kernel_size=kernel_size,
                                                  scope=sc+'/dw_conv',
                                                  reuse=tf.AUTO_REUSE)
    if(act_max[2*layer_no]>0):
      depthwise_conv = tf.fake_quant_with_min_max_vars(depthwise_conv, 
          min=-act_max[2*layer_no], 
          max=act_max[2*layer_no]-(act_max[2*layer_no]/128.0), 
          num_bits=8, name='quant_ds_conv'+str(layer_no))
    bn = tf.nn.relu(depthwise_conv)

    # batch-norm weights folded into depthwise conv 
    # bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_conv/batch_norm')
    pointwise_conv = slim.convolution2d(bn,
                                        num_pwc_filters,
                                        kernel_size=[1, 1],
                                        scope=sc+'/pw_conv',
                                        reuse=tf.AUTO_REUSE)
    if(act_max[2*layer_no+1]>0):
      pointwise_conv = tf.fake_quant_with_min_max_vars(pointwise_conv, 
          min=-act_max[2*layer_no+1], 
          max=act_max[2*layer_no+1]-(act_max[2*layer_no+1]/128.0), 
          num_bits=8, name='quant_pw_conv'+str(layer_no+1))
    bn = tf.nn.relu(pointwise_conv)

    # batch-norm weights folded into pointwise conv 
    # bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_conv/batch_norm')
    return bn

  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

  label_count = model_settings['label_count']
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
 
  t_dim = input_time_size
  f_dim = input_frequency_size

  #Extract model dimensions from model_size_info
  num_layers = model_size_info[0]
  conv_feat = [None]*num_layers
  conv_kt = [None]*num_layers
  conv_kf = [None]*num_layers
  conv_st = [None]*num_layers
  conv_sf = [None]*num_layers
  i=1
  for layer_no in range(0,num_layers):
    conv_feat[layer_no] = model_size_info[i]
    i += 1
    conv_kt[layer_no] = model_size_info[i]
    i += 1
    conv_kf[layer_no] = model_size_info[i]
    i += 1
    conv_st[layer_no] = model_size_info[i]
    i += 1
    conv_sf[layer_no] = model_size_info[i]
    i += 1

  scope = 'DS-CNN'
  with tf.variable_scope(scope) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                        activation_fn=None,
                        weights_initializer=slim.initializers.xavier_initializer(),
                        biases_initializer=slim.init_ops.zeros_initializer(),
                        outputs_collections=[end_points_collection]):
      with slim.arg_scope([slim.batch_norm],
                          is_training=is_training,
                          decay=0.96,
                          updates_collections=None,
                          activation_fn=tf.nn.relu):
        if act_max[0]>0:
          fingerprint_4d = tf.fake_quant_with_min_max_vars(fingerprint_4d, 
              min=-act_max[0], max=act_max[0]-(act_max[0]/128.0), 
              num_bits=8, name='quant_input')
        for layer_no in range(0,num_layers):
          if layer_no==0:
            net = slim.convolution2d(fingerprint_4d, conv_feat[layer_no],\
                [conv_kt[layer_no], conv_kf[layer_no]], stride=[conv_st[layer_no], 
                 conv_sf[layer_no]], padding='SAME', scope='conv_1', reuse=tf.AUTO_REUSE)
            if act_max[1]>0:
              net = tf.fake_quant_with_min_max_vars(net, min=-act_max[1], 
                  max=act_max[1]-(act_max[1]/128.0), num_bits=8, name='quant_conv1')
            net = tf.nn.relu(net)
            #net = slim.batch_norm(net, scope='conv_1/batch_norm')
          else:
            net = _depthwise_separable_conv(net, conv_feat[layer_no], \
                      kernel_size = [conv_kt[layer_no],conv_kf[layer_no]], \
                      stride = [conv_st[layer_no],conv_sf[layer_no]], 
                      sc='conv_ds_'+str(layer_no),
                      layer_no = layer_no,
                      act_max = act_max)
          t_dim = math.ceil(t_dim/float(conv_st[layer_no]))
          f_dim = math.ceil(f_dim/float(conv_sf[layer_no]))

        net = slim.avg_pool2d(net, [t_dim, f_dim], scope='avg_pool')
        if act_max[2*num_layers]>0:
          net = tf.fake_quant_with_min_max_vars(net, min=-act_max[2*num_layers], 
                    max=act_max[2*num_layers]-(act_max[2*num_layers]/128.0), 
                    num_bits=8, name='quant_pool')

    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
    logits = slim.fully_connected(net, label_count, activation_fn=None, 
                 scope='fc1', reuse=tf.AUTO_REUSE)
    if act_max[2*num_layers+1]>0:
      logits = tf.fake_quant_with_min_max_vars(logits, min=-act_max[2*num_layers+1], 
                   max=act_max[2*num_layers+1]-(act_max[2*num_layers+1]/128.0), 
                   num_bits=8, name='quant_fc')


  if is_training:
    return logits, dropout_prob
  else:
    return logits

def fake_quant(inputs, clamp, num_bits):
  max_abs = 2**(num_bits-1) * 1.0
  return tf.fake_quant_with_min_max_vars(inputs, -clamp, clamp-clamp/max_abs, num_bits=num_bits)

GATE_RANGE=8
NUM_BITS=8

class BasicLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
  def call(self, inputs, state):
    """Long short-term memory cell (LSTM).

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size, num_units]`, if `state_is_tuple` has been set to
        `True`.  Otherwise, a `Tensor` shaped
        `[batch_size, 2 * num_units]`.

    Returns:
      A pair containing the new hidden state, and the new state (either a
        `LSTMStateTuple` or a concatenated state, depending on
        `state_is_tuple`).
    """
    sigmoid = math_ops.sigmoid
    one = constant_op.constant(1, dtype=dtypes.int32)
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

    # force quantizing params (no effects?)
    self._kernel = fake_quant(self._kernel, 1, NUM_BITS)
    self._bias = fake_quant(self._bias, 1, NUM_BITS)

    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, h], 1), self._kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

    # tf.summary.histogram('gate', gate_inputs) # works
    # tf.summary.histogram('c_state', c) # works

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(
        value=gate_inputs, num_or_size_splits=4, axis=one)

    i = fake_quant(i, GATE_RANGE, NUM_BITS)
    j = fake_quant(j, GATE_RANGE, NUM_BITS)
    f = fake_quant(f, GATE_RANGE, NUM_BITS)
    o = fake_quant(o, GATE_RANGE, NUM_BITS)

    forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    add = math_ops.add
    multiply = math_ops.multiply
    # new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
    #             multiply(sigmoid(i), self._activation(j)))
    # new_h = multiply(self._activation(new_c), sigmoid(o))

    f_sig = fake_quant(sigmoid(add(f, forget_bias_tensor)), GATE_RANGE, NUM_BITS)

    new_c_1 = fake_quant(multiply(c, f_sig), GATE_RANGE, NUM_BITS)

    i_sig = fake_quant(sigmoid(i), GATE_RANGE, NUM_BITS)

    j_act = fake_quant(self._activation(j), GATE_RANGE, NUM_BITS)

    new_c_2 = fake_quant(multiply(i_sig, j_act), GATE_RANGE, NUM_BITS)

    new_c = fake_quant(add(new_c_1, new_c_2), GATE_RANGE, NUM_BITS)

    o_sig = fake_quant(sigmoid(o), GATE_RANGE, NUM_BITS)

    c_act = fake_quant(self._activation(new_c), GATE_RANGE, NUM_BITS)

    new_h = fake_quant(multiply(c_act, o_sig), GATE_RANGE, NUM_BITS)
    if self._state_is_tuple:
      new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)
    return new_h, new_state    

def create_basic_lstm_model(fingerprint_input, model_settings, model_size_info, 
                              act_max, is_training):
  """Builds a model with a basic lstm layer (without output projection and 
       peep-hole connections)
  model_size_info: defines the number of memory cells in basic lstm model
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size])
  
  flow = fake_quant(fingerprint_4d, 16, 8)  
  flow = tf.unstack(flow, input_time_size, axis=1) # required by static_rnn

  num_classes = model_settings['label_count']
  # flow = fingerprint_4d
  print(".........................")
  print(model_size_info, type(model_size_info))
  if type(model_size_info) is list:
    LSTM_units = model_size_info
  else:
    LSTM_units = [model_size_info] 
  num_layers = len(model_size_info)  
  # tf.summary.histogram('fingerprints', flow)

  with tf.name_scope('LSTM-Layer'):
    for i in range(num_layers):
      lstmcell = BasicLSTMCell(LSTM_units[i], forget_bias=0)
      flow, [_, flow_t]= tf.nn.static_rnn(cell=lstmcell, inputs=flow, dtype=tf.float32,scope='lstm'+str(i))
      # tf.summary.histogram('h_state', flow)
      # flow_t = fake_quant(flow_t, act_max[i+1], 8)

  with tf.name_scope('Output-Layer'):
    W_o = tf.get_variable('W_o', shape=[LSTM_units[-1], num_classes], 
            initializer=tf.contrib.layers.xavier_initializer())
    b_o = tf.get_variable('b_o', shape=[num_classes])
    logits = tf.matmul(flow_t, W_o) + b_o
    # tf.summary.histogram('logits', logits)

    logits = fake_quant(logits, 8, 8)

  if is_training:
    return logits, dropout_prob
  else:
    return logits

class LSTMCell(tf.nn.rnn_cell.LSTMCell):
  def call(self, inputs, state, gate_range=8, proj_bits=4, num_bits=4):
    """ quantized version of lstm cell with projection layer and w/o peephole
    """
    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = math_ops.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    # c_prev = fake_quant(c_prev, gate_range, num_bits)
    # m_prev = fake_quant(m_prev, gate_range, proj_bits)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    self._kernel = fake_quant(self._kernel, 1, 4)
    self._bias = fake_quant(self._bias, 1, 4)

    lstm_matrix = math_ops.matmul(
        array_ops.concat([inputs, m_prev], 1), self._kernel)
    lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

    # check internal value range
    # tf.summary.histogram('gate', lstm_matrix)
    # tf.summary.histogram('c_state', c_prev)
    # tf.summary.histogram('m_state', m_prev)

    i, j, f, o = array_ops.split(
        value=lstm_matrix, num_or_size_splits=4, axis=1)

    i = fake_quant(i, gate_range, num_bits)
    j = fake_quant(j, gate_range, num_bits)
    f = fake_quant(f, gate_range, num_bits)
    o = fake_quant(o, gate_range, num_bits)

    # Diagonal connections
    fsig = fake_quant(sigmoid(f + self._forget_bias), 1, num_bits) 
    c1 = fake_quant(fsig * c_prev, 1, num_bits)
    isig = fake_quant(sigmoid(i), 1, num_bits) 
    c2 = fake_quant(isig * self._activation(j), 1, num_bits)
    c = c1 + c2
    c = fake_quant(c, 1, num_bits)

    # TODO: test effects of clipping
    if self._cell_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
      # pylint: enable=invalid-unary-operand-type

    osig = fake_quant(sigmoid(o), 1, num_bits) 
    m = fake_quant(osig * self._activation(c), 1, proj_bits)

    m = math_ops.matmul(m, self._proj_kernel)
    m = fake_quant(m, 1, num_bits)

    if self._proj_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
      # pylint: enable=invalid-unary-operand-type

    new_state = (tf.nn.rnn_cell.LSTMStateTuple(c, m) if self._state_is_tuple else
                 array_ops.concat([c, m], 1))
    return m, new_state

def create_lstm_model(fingerprint_input, model_settings, model_size_info, act_max,
                        is_training):
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size])
  # tf.summary.histogram('fingerprint_4d', fingerprint_4d)

  flow = fake_quant(fingerprint_4d, act_max[0], 8)
  flow = tf.unstack(flow, input_time_size, axis=1) # required by static_rnn
  num_classes = model_settings['label_count']
  projection_units = model_size_info[0]
  LSTM_units = model_size_info[1]
  num_layers = int(len(model_size_info)/2)
  # print("number of layers: ", num_layers)
  with tf.name_scope('LSTM-Layer'):
    with tf.variable_scope("lstm"): 
      for i in range(num_layers):
        lstmcell = LSTMCell(LSTM_units, forget_bias=0,num_proj=projection_units, cell_clip = 1.0, proj_clip = 1.0)
        flow, [_, flow_t]= tf.nn.static_rnn(cell=lstmcell, inputs=flow, dtype=tf.float32,scope='lstm'+str(i))

  with tf.name_scope('Output-Layer'):
    W_o = tf.get_variable('W_o', shape=[projection_units, num_classes], 
            initializer=tf.contrib.layers.xavier_initializer())
    b_o = tf.get_variable('b_o', shape=[num_classes])
    W_o = fake_quant(W_o, 1, 4)
    b_o = fake_quant(b_o, 1, 4)
    logits = tf.matmul(flow_t, W_o) + b_o
    # logits = fake_quant(logits, act_max[num_layers+1], 8)
    logits = fake_quant(logits, 8, 4)
  if is_training:
    return logits, dropout_prob
  else:
    return logits

def create_crnn_model(fingerprint_input, model_settings, 
                                  model_size_info, act_max, is_training):
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  flow = fake_quant(fingerprint_4d, act_max[0], 8)    

  # CNN part
  first_filter_count = model_size_info[0]
  first_filter_height = model_size_info[1]
  first_filter_width = model_size_info[2]
  first_filter_stride_y = model_size_info[3]
  first_filter_stride_x = model_size_info[4]

  first_weights = tf.get_variable('W', shape=[first_filter_height, 
                    first_filter_width, 1, first_filter_count])

  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(flow, first_weights, [
      1, first_filter_stride_y, first_filter_stride_x, 1
  ], 'VALID') + first_bias
  first_relu = tf.nn.relu(first_conv)

  fake_quant(first_relu, 16, 8)
  tf.summary.histogram('relu_out', first_relu) 

  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  first_conv_output_width = int(math.floor(
      (input_frequency_size - first_filter_width + first_filter_stride_x) /
      first_filter_stride_x))
  first_conv_output_height = int(math.floor(
      (input_time_size - first_filter_height + first_filter_stride_y) /
      first_filter_stride_y))

  num_rnn_layers = model_size_info[5]
  RNN_units = model_size_info[6]
  flow = tf.reshape(first_dropout, [-1, first_conv_output_height, 
           first_conv_output_width * first_filter_count])
  flow = tf.unstack(flow, first_conv_output_height, axis=1) # required by static_rnn

  for i in range(num_rnn_layers):
    lstmcell = BasicLSTMCell(RNN_units, forget_bias=0)
    flow, [_, flow_t]= tf.nn.static_rnn(cell=lstmcell, inputs=flow, dtype=tf.float32,scope='lstm'+str(i))
  # Quant. done in LSTMCell
  tf.summary.histogram('all_hstates', flow) 

  flow_dim = RNN_units
  first_fc_output_channels = model_size_info[7]
  first_fc_weights = tf.get_variable('fcw', shape=[flow_dim, 
    first_fc_output_channels])
  
  first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
  first_fc = tf.nn.relu(tf.matmul(flow_t, first_fc_weights) + first_fc_bias)

  fake_quant(first_fc, 8, 4)
  tf.summary.histogram('first_fc', first_fc) 

  if is_training:
    final_fc_input = tf.nn.dropout(first_fc, dropout_prob)
  else:
    final_fc_input = first_fc

  label_count = model_settings['label_count']
  
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [first_fc_output_channels, label_count], stddev=0.01))

  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias

  fake_quant(first_fc, 16, 4)  
  tf.summary.histogram('final_fc', final_fc) 

  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc    