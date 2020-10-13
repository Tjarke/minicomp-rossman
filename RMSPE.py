import numpy as np
import pandas as pd

def RMSPE(y_true, y_pred):
    """
    Calculate the root mean square percentage error

    INPUT:
    y_true: true values for the target variable
    y_pred: predicted values for the target variable

    OUTPUT:
    Value between 0 and 1, where 0 is the best and
    """
    y_difference = np.square((y_true - y_pred)/y_true)
    y_difference_df = pd.DataFrame(y_difference)
    y_difference = y_difference_df[~(y_difference_df[0] == np.inf)]
    RMSPE = np.sqrt(np.mean(y_difference))
    return RMSPE
