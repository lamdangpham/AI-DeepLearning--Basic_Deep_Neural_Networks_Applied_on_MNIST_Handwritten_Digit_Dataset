import tensorflow as tf
import numpy as np
import os

from model_para    import *
from dnn_bl01_conf import *

#======================================================================================================#

class model_conf(object):

    def __init__( self):

        self.model_para = model_para()

        # ============================== Fed Input
        #input data
        self.input_layer_val  = tf.placeholder(tf.float32, [None, self.model_para.n_dim], name="input_layer_val")
        
        #expected class
        self.expected_classes = tf.placeholder(tf.float32, [None, self.model_para.n_class], name="expected_classes")

        #run mode
        self.mode             = tf.placeholder(tf.bool, name="running_mode")


        #============================== NETWORK CONFIGURATION
        # Call DNN 
        with tf.device('/gpu:0'), tf.variable_scope("dnn_01")as scope:
            self.dnn_bl01_ins_01   = dnn_bl01_conf(self.input_layer_val, self.model_para.n_dim, self.mode)
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
