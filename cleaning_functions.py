import pandas as pd
import numpy as np


#Extract the days the store is closed from the dataset!
#Input dataframe
#Output: dataframe with closed days, dataframe with open days



def extract_closed(df):
    select = df.loc[:,"Open"] == 0
    return df.loc[select,:],df.loc[~select,:]