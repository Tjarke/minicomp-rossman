import numpy as np
import pandas as pd

def fit_sales_baseline0(Train_data):
    """
    This function fit the model to predict the sales based on the average sales per store of the las 1.5 years

    INPUT:
    - Train_data: Dataset for training

    OUTPUT:
    - Dictionary with the average sales per store
    """

    mean_per_store_dict = Train_data.groupby(['Store']).Sales.mean().to_dict()
    return mean_per_store_dict

def predict_sales_baseline0(Test_data, model_fit):
    """
    This function predicts the sales based on the store. It uses the average sales of the last 1.5 years

    INPUT:
    X_matrix: data with all the relevant features: 'Date', 'Store', 'DayOfWeek', 'Open', 'Promo',
              'StateHoliday' and 'SchoolHoliday'
    OUTPUT:
    prediction: a y-vector with the predictions
    """

    y = Test_data['Store'].map(model_fit)
    return y
