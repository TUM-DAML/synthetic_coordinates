import numpy as np
from sklearn.metrics import mean_absolute_error

class ZINC_Evaluator:
    def __init__(self):
        pass

    def eval(self, input_dict):
        y_true = input_dict['y_true']
        y_pred = input_dict['y_pred']
    
        return {'mae': mean_absolute_error(y_pred, y_true)}

class QM9_Evaluator:
    def __init__(self):
        pass

    def eval(self, input_dict):
        y_true = input_dict['y_true']
        y_pred = input_dict['y_pred']
    
        return {'mae': mean_absolute_error(y_pred, y_true, multioutput='raw_values')[0]}