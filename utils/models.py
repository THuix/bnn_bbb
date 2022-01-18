from model_linear import Linear_BNN
from model_conv import Conv_BNN
from model_vgg import VGG
import numpy as np

# Linear BNN - 1 hidden layer
class Linear_regime_1(Linear_BNN):
    def __init__(self, dist_params, train_params, model_params):
        super(Linear_regime_1, self).__init__(1, dist_params, train_params, model_params)

class Linear_regime_2(Linear_BNN):
    def __init__(self, dist_params, train_params, model_params):
        dist_params['sigma_prior'] =  dist_params['sigma_prior'] * np.sqrt(w / (train_params['alpha'] * p))
        super(Linear_regime_2, self).__init__(2, dist_params, train_params, model_params)

class Linear_regime_3(Linear_BNN):
    def __init__(self, dist_params, train_params, model_params):
        super(Linear_regime_3, self).__init__(3, dist_params, train_params, model_params)


# Conv BNN - 1 hidden layer
class Conv_regime_1(Conv_BNN):
    def __init__(self, dist_params, train_params, model_params):
        super(Conv_regime_1, self).__init__(1, dist_params, train_params, model_params)

class Conv_regime_3(Conv_BNN):
    def __init__(self, dist_params, train_params, model_params):
        super(Conv_regime_3, self).__init__(3, dist_params, train_params, model_params)

# VGG models
class VGG_regime_1(VGG):
    def __init__(self, dist_params, train_params, model_params):
        super(VGG_regime_1, self).__init__(1, dist_params, train_params, model_params)

class VGG_regime_3(VGG):
    def __init__(self, dist_params, train_params, model_params):
        super(VGG_regime_3, self).__init__(3, dist_params, train_params, model_params)



