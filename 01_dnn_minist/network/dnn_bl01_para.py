import numpy as np
import os

class dnn_bl01_para(object):
    """
    define a class to store parameters
    """

    def __init__(self):

        #=======Layer 01: full connection
        self.l01_fc             = 2048 
        self.l01_is_act         = True
        self.l01_act_func       = 'RELU'
        self.l01_is_drop        = True
        self.l01_drop_prob      = 0.2

        #=======Layer 02: full connection
        self.l02_fc             = 4096  
        self.l02_is_act         = True
        self.l02_act_func       = 'RELU'
        self.l02_is_drop        = True
        self.l02_drop_prob      = 0.2

        #=======Layer 03: full connection
        self.l03_fc             = 4096  
        self.l03_is_act         = True
        self.l03_act_func       = 'RELU'
        self.l03_is_drop        = True
        self.l03_drop_prob      = 0.2

        #=======Layer 04: full connection
        self.l04_fc             = 1024  
        self.l04_is_act         = True
        self.l04_act_func       = 'RELU'
        self.l04_is_drop        = True
        self.l04_drop_prob      = 0.2

        #=======Layer 05: Final layer
        self.l05_fc             = 10   
        self.l05_is_act         = False
        self.l05_act_func       = 'RELU'
        self.l05_is_drop        = False
        self.l05_drop_prob      = 1

