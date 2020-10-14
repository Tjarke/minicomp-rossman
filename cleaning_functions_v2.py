import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler



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



def sales_store(df1, df2):
    """
    get the average sales per store and save them in a new column in the DataFrame. The function returns a
    DataFrame
    INPUT:
    - df: DataFrame Train = df1 and Datframe Test = df2
    OUTPUT:
    - DataFrame with a new column with the average sales per store df1 = train and df2 = test
    """
    df1 = df1.copy()
    df2 = df2.copy()
    sales_store_dict = df1.groupby(['Store']).Sales.mean().to_dict()
    df1['sales_store'] = df1['Store'].map(sales_store_dict)
    df2['sales_store'] = df2['Store'].map(sales_store_dict)
    return df1, df2

def sales_customer_store(df1,df2):
    """
    Return a new column in the DataFrame with the average sales per customer per store
    INPUT:
    - df:  DataFrame Train = df1 and Datframe Test = df2
    OUTPUT:
    - DataFrame with a new column with the average sales per customer per store df1 = train and df2 = test
    """
    df1 = df1.copy()
    df2 = df2.copy()
    # create a column with the ratio Sales/Customer
    df1['sales_customer'] = df1['Sales']/df1['Customers']
    # create a dictionary with the average sales per customer per store
    sales_customer_store_dict = df1.groupby(['Store'])['sales_customer'].mean().to_dict()
    # create a new column with the average sales per customer per store
    df1['sales_customer_store'] = df1['Store'].map(sales_customer_store_dict)
    df2['sales_customer_store'] = df2['Store'].map(sales_customer_store_dict)
    # drop the sales_customer column
    df1.drop(columns=['sales_customer'], inplace=True)
    return df1,df2






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
    #Drop unnecesary columns:
    df.drop(columns = ['Open', 'StateHoliday',
                   'SchoolHoliday',
                   'CompetitionOpenSinceMonth',
                   'CompetitionOpenSinceYear',
                   'Promo2SinceWeek',
                   'Promo2SinceYear'], inplace=True)



    #we fill  df['PromoInterval'] so that we dont drop rows in the line below


    df['PromoInterval'] = df['PromoInterval'].fillna(0)

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






    #split into train and validation Set

    train_size = 0.8

    X_train = df.iloc[:round(train_size*df.shape[0]),:]
    X_test = df.iloc[round(train_size*df.shape[0]):,:]



    scaler = StandardScaler()
    X_train.loc[:,"CompetitionDistance"] = scaler.fit_transform(X_train.loc[:,"CompetitionDistance"].to_numpy().reshape(-1, 1))
    X_test.loc[:,"CompetitionDistance"] = scaler.transform(X_test.loc[:,"CompetitionDistance"].to_numpy().reshape(-1, 1))


#    X_train, X_test = sales_store(X_train,X_test)
    X_train, X_test = sales_customer_store(X_train,X_test)

    # change week to type integer since it is saved in a different way
    X_train['Week'] = X_train['Week'].astype(int)
    X_test['Week'] = X_test['Week'].astype(int)

    X_train.drop(columns = ["Sales", "Customers", "Store"], inplace = True)
    X_test.drop(columns = ["Sales", "Customers", "Store"], inplace = True)
    y_train = df.iloc[:round(train_size*df.shape[0]),:].loc[:,"Sales"]
    y_test = df.iloc[round(train_size*df.shape[0]):,:].loc[:,"Sales"]



    return X_train,y_train,X_test,y_test
