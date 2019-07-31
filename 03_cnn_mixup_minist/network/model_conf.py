import tensorflow as tf
import numpy as np
import os

from model_para    import *
from cnn_bl_conf   import *
from dnn_bl01_conf import *

#======================================================================================================#

class model_conf(object):

    def __init__( self):

        self.model_para = model_para()

        # ============================== Fed Input
        #input data
        self.input_layer_val  = tf.placeholder(tf.float32, [None, self.model_para.n_hig, self.model_para.n_wid, self.model_para.n_chan], name="input_layer_val")
        
        #expected class
        self.expected_classes = tf.placeholder(tf.float32, [None, self.model_para.n_class], name="expected_classes")

        #run mode
        self.mode             = tf.placeholder(tf.bool, name="running_mode")

        #============================== NETWORK CONFIGURATION

        # Call CNN 
        with tf.device('/gpu:0'), tf.variable_scope("cnn_01")as scope:
            self.cnn_ins_01 = cnn_bl_conf(self.input_layer_val, self.mode)
            [nBatch, nDim]  = self.cnn_ins_01.final_output.get_shape()

        # Call DNN 
        with tf.device('/gpu:0'), tf.variable_scope("dnn_01")as scope:
            self.dnn_bl01_ins_01   = dnn_bl01_conf(self.cnn_ins_01.final_output, int(nDim), self.mode)
            self.output_layer      = self.dnn_bl01_ins_01.final_output
            self.prob_output_layer = tf.nn.softmax(self.dnn_bl01_ins_01.final_output)

        ### ======================================== LOSS FUNCTION AND ACCURACY =========================
        ### loss function
        with tf.device('/gpu:0'), tf.variable_scope("loss") as scope:
  
            # l2 loss  
            l2_loss = self.model_para.l2_lamda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

            # main loss
            dummy = tf.constant(0.00001) # to avoid dividing by 0 with KL divergence
            p = self.expected_classes  + dummy
            q = self.prob_output_layer + dummy
            losses_mix = tf.reduce_sum(p * tf.log(p/q))

            # final loss
            self.loss =  losses_mix + l2_loss

        ### Calculate Accuracy  
        with tf.device('/gpu:0'), tf.name_scope("accuracy") as scope:
            self.correct_prediction = tf.equal(tf.argmax(self.output_layer,1), tf.argmax(self.expected_classes,1))
            self.accuracy           = tf.reduce_mean(tf.cast(self.correct_prediction,"float", name="accuracy" ))
