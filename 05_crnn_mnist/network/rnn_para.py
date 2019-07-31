# import tensorflow as tf
import numpy as np
import os


class rnn_para(object):

    def __init__(self):

        self.input_drop   = 1
        self.output_drop  = 1
        self.n_layer      = 1
        self.n_hidden     = 28 
        self.nframe       = 28  #TODO need to match with step02...

