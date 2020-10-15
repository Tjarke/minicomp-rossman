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
    select = df.loc[:,"Open"] == 0
    return df.loc[~select,:],df.loc[select,:]


#input: both data frames
#output: X_train,y_train,X_test,y_test

def clean_complete(test_data, store_data):

    # make a copy of the dataset for cleaning purposes
    df1 = test_data.copy()
    df2 = store_data.copy()

    # merge both datasets
    df = pd.merge(df1, df2, how='inner', on=['Store'], sort=False, suffixes=('_test', '_store'), copy=True)

    # set Open's NaN values to 1 when sales where > 0
    select = (df.loc[:,'Open'].isnull()) & (df.loc[:,'Sales'] > 0)
    df.loc[select, 'Open'] = 1

    # get the day, week, and month
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek + 1
    # make sure the variables have the right datatype for the modeling part
    df['Week'] = df['Week'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df['DayOfWeek'] = df['DayOfWeek'].astype(int)

    #we fill  df['PromoInterval'] so that we dont drop rows in the line below
    df['PromoInterval'] = df['PromoInterval'].fillna(0)
    # create Month and Week columns. Furthermore, we create a new DayOfWeek column
    # to get rid of the NaN


    print('before droping the NaN, the test set has a shape of {}'.format(df.shape))
    #drop nas
     df = df.dropna(axis=0)
    print('after droping the NaN, the test set has a shape of {}'.format(df.shape))


    # sort the values to get original set
    df = df.sort_values(by = "Date")

    ############ ENCODING ############
    # label encode for Assortment
    le = LabelEncoder()  #instantiate the Label Encoder
    df.loc[:,'Assortment'] = le.fit_transform(df.loc[:,'Assortment'])

    # one hot encode for store type
    ce_one = ce.OneHotEncoder(cols=['StoreType'])
    df = ce_one.fit_transform(df)

    # Encode Promo2 according to Promointerval
    select_jan_etc = (df['PromoInterval'] == "Jan,Apr,Jul,Oct") & (df.loc[:,"Month"].isin([1,4,7,10]))
    select_feb_etc = (df['PromoInterval'] == "Feb,May,Aug,Nov") & (df.loc[:,"Month"].isin([2,5,8,11]))
    select_mar_etc = (df['PromoInterval'] == "Mar,Jun,Sept,Dec") & (df.loc[:,"Month"].isin([3,6,9,12]))

    df.loc[:,"Promo2"]=0
    df.loc[select_jan_etc,"Promo2"] = 1
    df.loc[select_feb_etc,"Promo2"] = 1
    df.loc[select_mar_etc,"Promo2"] = 1

    # MeanEncoding for StoreID - we replace it with the variable sales per customers per store
    # Import trained dictionary
    with open ('sales_customer_store_dict.txt', 'rb') as myFile:
        sales_customer_store_dict = pickle.load(myFile)

    # map the dictionary to each store
    df['sales_customer_store'] = df['Store'].map(sales_customer_store_dict)


    print('After encoding the shape of the dataset is {}, remember that \nno columns have been dropped so far'.format(df.shape))

    ########## STANDARDIZATION #############
    scaler = StandardScaler()
    ### IMPORTATN import pickle with scaler!
    # df.loc[:,"CompetitionDistance"] = scaler.transform(df.loc[:,"CompetitionDistance"].to_numpy().reshape(-1, 1))

    # get the clean data
    df_clean = df[['DayOfWeek',
                   'Promo',
                   'StoreType_1',
                   'StoreType_2',
                   'StoreType_3',
                   'StoreType_4',
                   'Assortment',
                   'CompetitionDistance',
                   'Promo2',
                   'Month',
                   'Week',
                   'sales_customer_store']].copy()

    return df_clean

    #
    # #Drop unnecesary columns:
    # df.drop(columns = ['StateHoliday',
    #                'SchoolHoliday',
    #                'CompetitionOpenSinceMonth',
    #                'CompetitionOpenSinceYear',
    #                'Promo2SinceWeek',
    #                'Promo2SinceYear'], inplace=True)
    # df.drop(columns = ["Date"], inplace = True)
    # df.drop(columns = ["PromoInterval"],inplace = True)
    #
