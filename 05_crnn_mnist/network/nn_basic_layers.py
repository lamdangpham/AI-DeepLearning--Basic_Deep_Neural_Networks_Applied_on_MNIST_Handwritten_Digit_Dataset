import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
import tensorflow.contrib.layers as layers
from bnlstm import BNLSTMCell

"""
Predefine all necessary layer for the R-CNN
"""

def bidirectional_recurrent_layer(nhidden, nlayer, input_keep_prob=1.0, output_keep_prob=1.0):
    fw_cell = tf.contrib.rnn.GRUCell(nhidden)
    bw_cell = tf.contrib.rnn.GRUCell(nhidden)
    if (nlayer > 1):
        fw_cell = tf.contrib.rnn.MultiRNNCell([fw_cell] * nlayer,
                                              state_is_tuple=True
                                             )  # due to built with LN LSTM

        bw_cell = tf.contrib.rnn.MultiRNNCell([bw_cell] * nlayer,
                                              state_is_tuple=True
                                             )  # due to built with LN LSTM
    # input & output dropout
    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, input_keep_prob  = input_keep_prob)
    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob = output_keep_prob)
    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, input_keep_prob  = input_keep_prob)
    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob = output_keep_prob)

    return fw_cell,bw_cell

def bidirectional_recurrent_layer_output_new(fw_cell, bw_cell, input_layer, sequence_len, scope=None):
    ((fw_outputs,
      bw_outputs),
     (fw_state,
      bw_state)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                    cell_bw=bw_cell,
                                                    inputs=input_layer,
                                                    sequence_length=sequence_len,
                                                    dtype=tf.float32,
                                                    swap_memory=True,
                                                    scope=scope)
                                                   )

    outputs = tf.concat((fw_outputs, bw_outputs), 2)

    def concatenate_state(fw_state, bw_state):
        if isinstance(fw_state, LSTMStateTuple):
            state_c = tf.concat((fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
            state_h = tf.concat((fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
            state = LSTMStateTuple(c=state_c, h=state_h)
            return state
        elif isinstance(fw_state, tf.Tensor):
            state = tf.concat((fw_state, bw_state), 1,
                              name='bidirectional_concat')
            return state
        elif (isinstance(fw_state, tuple) and isinstance(bw_state, tuple) and len(fw_state) == len(bw_state)):
            # multilayer
            state = tuple(concatenate_state(fw, bw) for fw, bw in zip(fw_state, bw_state))
            return state

        else:
            raise ValueError('unknown state type: {}'.format((fw_state, bw_state)))

    state = concatenate_state(fw_state, bw_state)

    return outputs, state

