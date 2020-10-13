import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder




#Extract the days the store is closed from the dataset!
#Input dataframe
#Output: dataframe with open days, dataframe with closed days
def extract_closed(df):
    select = df.loc[:,"Open"] == 0
    return df.loc[~select,:],df.loc[select,:]


        

def no_outlier(df1, col):
    """
    This function get rid of outliers for variables that have a very skewed data, such
    that a log transformation approximates a normal distribution. It identifies all datapoints
    that are 3 standard deviations away from the mean as outliers and remove this datapoints
    from the DataFrame
    IMPORTANT:
    - The outlier detection is based on the assumption of a normal distribution
    if this is not your case, then use another outlier detection method
    - Because the outlier detection does a log transformation first, make sure the data
    contians no zeros
    INPUT:
    - df1: DataFrame
    - col: Column (variable) in the df, where outliers have to be removed
    OUTPUT: DataFrame and selected column without outliers
    """
    df = df1.copy()
    df[col] = np.log(df[col]) # apply a log transformation to the var
    mean = np.mean(df[col])
    std = np.std(df[col])
    df[col] = ((df[col] - mean) / std) # apply standardization
    df = df.drop(df[(df[col] > 3) ^ (df[col] < -3)].index) # get rid of datapoints
                                                            # 3 stds away from the mean
    df[col] = ((df[col]*std) + mean) # reverse the standardization
    df[col] = np.exp(df[col]) # reverse the log transformation
    return df








#input: both data frames 
#output: X_train,y_train,X_test,y_test

def clean_complete(df1,df2):
    
    #merge dfs
    df = pd.merge(df2, df1, how='inner', on=['Store'], sort=False, suffixes=('_train', '_store'), copy=True)
    # df = pd.merge(df2, df1, how='inner', on = ["Store"],
    #      left_index=False, right_index=False, sort="Date",
    #      suffixes=('_x', '_y'), copy=True, indicator=False,
    #      validate=None)
    
    
    
    
    
    # extract closed days
    df,unused = extract_closed(df)
    
    df.drop(columns = ['StateHoliday',
                   'SchoolHoliday',
                   'CompetitionOpenSinceMonth',
                   'CompetitionOpenSinceYear',
                   'Promo2',
                   'Promo2SinceWeek',
                   'Promo2SinceYear',
                   'PromoInterval'], inplace=True)
    
    
    
    #drop nas
    df = df.dropna(axis=0)
    
    
    #label encode for Assortment
    
    le = LabelEncoder()  #instantiate the Label Encoder
    df.loc[:,'Assortment'] = le.fit_transform(df.loc[:,'Assortment'])
    
    
    
    # one hot encode for store type
    ce_one = ce.OneHotEncoder(cols=['StoreType']) 
    
    df = ce_one.fit_transform(df)
    
    
    
    #get month and weeks as column
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    # df.drop(columns = ['Date'], inplace = True)
    
    
    
    #sort the values to get original set
    df = df.sort_values(by = "Date")
    df.drop(columns = "Date", inplace = True)
    

    #remove sales outliers, zeros as well as more than 3std deviations
    df = df[df.Sales != 0]
    df = no_outlier(df, "Sales")

    
    
    
    #split into train and validation Set
    
    test_size = 0.2
    
    X_train = df.iloc[:round(test_size*df.shape[0]),:]
    X_test = df.iloc[round(test_size*df.shape[0]):,:]
    y_train = df.iloc[:round(test_size*df.shape[0]),:].loc[:,"Sales"]
    y_test = df.iloc[round(test_size*df.shape[0]):,:].loc[:,"Sales"]
    
    
    return X_train,y_train,X_test,y_test

    


