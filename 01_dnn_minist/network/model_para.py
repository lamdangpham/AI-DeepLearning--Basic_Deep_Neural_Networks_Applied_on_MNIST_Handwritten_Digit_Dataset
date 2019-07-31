class model_para(object):
    """
    define a class to store parameters
    """

    def __init__(self):

        #======================= Trainging parameters
        self.n_class            = 10  # Final output classes 
        self.l2_lamda           = 0.0001  # lamda prarameter

        #========================  Input parameters
        self.n_dim              = 784  
