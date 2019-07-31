import numpy as np
import os

class dnn_bl01_para(object):
    """
    define a class to store parameters
    """

    def __init__(self):

        #=======Layer 05: full connection
        self.l01_fc             = 1024 
        self.l01_is_act         = True
        self.l01_act_func       = 'RELU'
        self.l01_is_drop        = True
        self.l01_drop_prob      = 0.2

        #=======Layer 07: Final layer
        self.l02_fc             = 10   
        self.l02_is_act         = False
        self.l02_act_func       = 'RELU'
        self.l02_is_drop        = False
        self.l02_drop_prob      = 1

