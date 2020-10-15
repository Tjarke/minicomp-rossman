import category_encoders as ce
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

#Extract the days the store is closed from the dataset!
#Input dataframe
#Output: dataframe with open days, dataframe with closed days
def extract_closed(df):
    """
    Extract the days where the store is closed from the dataset!

    Input Parameters
    ----------
    df : pandas dataframe 
        
    Returns
    -------
    output1 : dataframe containing rows with open != 0, meaning rows where the store was open
        
    output2 : dataframe containing rows with open == 0, meaning rows where the store was closed

    """
    
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




def sales_customer_store(df1):
    """
    Return a new column in the DataFrame with the average sales per customer per store
    INPUT:
    - df:  DataFrame Train = df1
    OUTPUT:
    output 1 :
    - DataFrame with a new column with the average sales per customer per store df1 = train 
    output 2:
    - Dictionary containing the link between store id and the corresponding sales per customers in this store
    
    
    """
    df1 = df1.copy()
    # create a column with the ratio Sales/Customer
    df1['sales_customer'] = df1['Sales']/df1['Customers']
    # create a dictionary with the average sales per customer per store
    sales_customer_store_dict = df1.groupby(['Store'])['sales_customer'].mean().to_dict()
    # create a new column with the average sales per customer per store
    df1['sales_customer_store'] = df1['Store'].map(sales_customer_store_dict)
    # drop the sales_customer column
    df1.drop(columns=['sales_customer'], inplace=True)
    return df1,sales_customer_store_dict




def predict_customers(df):
    """
    
    input:
        Dataframe containing Nans for "Customers" where sales > 0 
        
    Output:
        Dataframe containing a prediction for the customers based on the sales day of the week and store

    """
    df_customers = df[['Store', 'DayOfWeek', 'Sales', 'Customers']].copy()
    df_customers = df[(~df['Customers'].isnull()) & (~df['Sales'].isnull())].copy()
    # Import model predictor using pickle
    filename = 'customer_pred_model.sav'
    xgb_model = pickle.load(open(filename, 'rb'))
    # predict the values for customers
    select = (df.loc[:,'Customers'].isnull()) & (~df['Sales'].isnull())
    customer_pred = xgb_model.predict(df.loc[select, ['Store', 'DayOfWeek', 'Sales']])
    # fill in the values into the DataFrame
    df.loc[select, 'Customers'] = customer_pred.reshape(-1,1)
    return df

#input: both data frames
#output: X_train,y_train,X_test,y_test

def clean_complete(df1,df2):
    
    """
    This function will take as input the uncleaned train dataset and the store data set and ouput a feature dataset with a corresponding label dataset
    It will also produce a pickle file, containing the necessary dictionaries and models, so that proper encoding is possible on the test set
    
    
    Input:
        df1 = pandas dataset train
        df2 = pandas dataset store
    
    Outout:
        X_train = pandas dataset containing features used to train
        y_train = pandas dataset containing labels used to train
        it also saves a pickle file!
    """
    
    
    
    #merge dfs
    df = pd.merge(df2, df1, how='inner', on=['Store'], sort=False, suffixes=('_train', '_store'), copy=True)


    # set Open's NaN values to 1 when sales where > 0
    select = (df.loc[:,'Open'].isnull()) & (df.loc[:,'Sales'] > 0)
    df.loc[select, 'Open'] = 1

    # get the day, week, and month
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek + 1

    # predict all Customer's NaN values where sales > 0
    df = predict_customers(df)

    # extract closed days
    df,unused = extract_closed(df)
    #Drop unnecesary columns:
    df.drop(columns = ['Open', 'StateHoliday',
                   'SchoolHoliday',
                   'CompetitionOpenSinceMonth',
                   'CompetitionOpenSinceYear',
                   'Promo2SinceWeek',
                   'Promo2SinceYear'], inplace=True)



    #we fill  df['PromoInterval'] so that we dont drop rows in the line below
    df['PromoInterval'] = df['PromoInterval'].fillna(0)
    # create Month and Week columns. Furthermore, we create a new DayOfWeek column
    # to get rid of the NaN

    #drop nas
    df = df.dropna(axis=0)


    #label encode for Assortment
    le = LabelEncoder()  #instantiate the Label Encoder
    le.fit(df.loc[:,'Assortment'])
    df.loc[:,'Assortment'] = le.transform(df.loc[:,'Assortment'])


    # one hot encode for store type
    ce_one = ce.OneHotEncoder(cols=['StoreType'])
    df = ce_one.fit_transform(df)


    #sort the values to get original set
    df = df.sort_values(by = "Date")
    df.drop(columns = ["Date"], inplace = True)


    #remove sales outliers, zeros as well as more than 3std deviations
    df = df[df.Sales != 0]
    df = no_outlier(df, "Sales")


    select_jan_etc = (df['PromoInterval'] == "Jan,Apr,Jul,Oct") & (df.loc[:,"Month"].isin([1,4,7,10]))
    select_feb_etc = (df['PromoInterval'] == "Feb,May,Aug,Nov") & (df.loc[:,"Month"].isin([2,5,8,11]))
    select_mar_etc = (df['PromoInterval'] == "Mar,Jun,Sept,Dec") & (df.loc[:,"Month"].isin([3,6,9,12]))


    df.loc[:,"Promo2"]=0
    df.loc[select_jan_etc,"Promo2"] = 1
    df.loc[select_feb_etc,"Promo2"] = 1
    df.loc[select_mar_etc,"Promo2"] = 1

    df.drop(columns = ["PromoInterval"],inplace = True)


    # X_train = df
    
    
    scaler = StandardScaler()
    scaler.fit(df.loc[:,"CompetitionDistance"].to_numpy().reshape(-1, 1))
    df.loc[:,"CompetitionDistance"] = scaler.transform(df.loc[:,"CompetitionDistance"].to_numpy().reshape(-1, 1))



    df, sales_customer_store_dict = sales_customer_store(df)


    # change week to type integer since it is saved in a different way
    df.loc[:,'Week'] = df.loc[:,'Week'].astype(int)



    X_train = df
    y_train = df.loc[:,"Sales"]
    
    
    X_train.drop(columns = ["Sales", "Customers", "Store","StoreType_4"], inplace = True)
  
    
    pickle_dump_var = [le, scaler , sales_customer_store_dict]
    
    with open('encoder_dictionary.pickle', 'wb') as handle:
        pickle.dump(pickle_dump_var, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    return X_train,y_train
