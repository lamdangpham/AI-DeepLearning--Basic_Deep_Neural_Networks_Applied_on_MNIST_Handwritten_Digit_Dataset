import tensorflow as tf
import numpy as np
import os

from model_para      import *
from rnn_para        import *

from dnn_bl01_conf   import *
from cnn_bl_conf     import *
from nn_basic_layers import *

#======================================================================================================#

class model_conf(object):

    def __init__( self):

        self.model_para = model_para()
        self.rnn_para = rnn_para()

        # ============================== Fed Input
        #input data
        self.input_layer_val  = tf.placeholder(tf.float32, [None, self.model_para.n_hig, self.model_para.n_wid], name="input_layer_val")
        
        #expected class
        self.expected_classes = tf.placeholder(tf.float32, [None, self.model_para.n_class], name="expected_classes")

        #run mode
        self.mode             = tf.placeholder(tf.bool, name="running_mode")

        #seq length
        self.seq_len          = tf.placeholder(tf.int32, name="sequence_length")


        #============================== NETWORK CONFIGURATION
        #CNN and RNN are parallel
        #Call CNN
        with tf.device('/gpu:0'), tf.variable_scope("cnn_01")as scope:
            self.i_cnn = tf.reshape(self.input_layer_val, shape=[-1, self.model_para.n_hig, self.model_para.n_wid, 1])
            self.cnn_ins_01 = cnn_bl_conf(self.i_cnn, self.mode)
            self.cnn_dim  = self.cnn_ins_01.final_output.get_shape()[1].value


        # Call RNN 
        with tf.device('/gpu:0'), tf.variable_scope("bidirection_recurrent_layer_01") as scope:
            self.fw_cell_01, self.bw_cell_01 = bidirectional_recurrent_layer(self.rnn_para.n_hidden,
                                                                             self.rnn_para.n_layer,
                                                                             input_keep_prob  = self.rnn_para.input_drop, 
                                                                             output_keep_prob = self.rnn_para.output_drop
                                                                            )

            self.rnn_out_01, self.rnn_state_01 = bidirectional_recurrent_layer_output_new(self.fw_cell_01,
                                                                                          self.bw_cell_01,
                                                                                          self.input_layer_val, #input of RNN
                                                                                          self.seq_len,
                                                                                          scope=scope
                                                                                         )
            self.rnn_state_dim = self.rnn_state_01.get_shape()[1].value

        # Call DNN 
        with tf.device('/gpu:0'), tf.variable_scope("dnn_01")as scope:
            self.i_dnn_dim = self.cnn_dim + self.rnn_state_dim
            self.i_dnn_data = tf.concat((self.cnn_ins_01.final_output, self.rnn_state_01), axis=1)

            self.dnn_bl01_ins_01   = dnn_bl01_conf(self.i_dnn_data, self.i_dnn_dim, self.mode)
            self.output_layer      = self.dnn_bl01_ins_01.final_output

        ### ======================================== LOSS FUNCTION AND ACCURACY =========================
        ### loss function
        with tf.device('/gpu:0'), tf.variable_scope("loss") as scope:
  
            # l2 loss  
            l2_loss = self.model_para.l2_lamda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

            # main loss
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.expected_classes, logits=self.output_layer)

            # final loss
            self.loss =  tf.reduce_mean(losses) + l2_loss

        ### Calculate Accuracy  
        with tf.device('/gpu:0'), tf.name_scope("accuracy") as scope:
            self.correct_prediction = tf.equal(tf.argmax(self.output_layer,1), tf.argmax(self.expected_classes,1))
            self.accuracy           = tf.reduce_mean(tf.cast(self.correct_prediction,"float", name="accuracy" ))
