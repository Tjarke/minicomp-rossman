# Rossman Kaggle Mini-Competition

This mini competition is adapted from the Kaggle Rossman challenge and was part of the Data Science Retreat Program 2020.
The Task is to predict future sales of Rossman-stores, using the data from two csv files:
- *store.csv*: general information for each store
- *train.csv*: daily sales information for each store

More information on the data in the Dataset section.

## Structure of the repository and how to use it
Make sure to read this and the next section before you start using the notebooks

### 1. The Results:
You will find our results and prediction accuracy in the **Rossman_sales_pred_notebook**. Here you can use our pre-trained model to get the root mean square percentage error (RMSPE) on the given Test-set. **How to use the notebook**:
  - Just run the cells one after another and you will get the RMSPE at the end. Please notice that this could take some time as the model we use is relatively complex. You can read more about the model in the Model section below.
  - The Notebook uses the python script 'Rossman_cleaning_pipeline' to clean the raw-data. You do not have to do/change anything here. The notebook makes everything for you.
  - Our trained model and some further steps are saved in the pickle file "encoder_dict.pkl". You do not have to do/change anything here. The notebook makes everything for you.


### 2. Setting up the model:
If you are interested in the setting-up the model and maybe tuning the model by yourself, please go into the folder 'setting_up_model'. Here you can follow step by step the cleaning and modeling process we went through. **Our goal is for you, to be able to reproduce our model and get the same results as we did**. However, you are free to play with the model and even try a new one for yourself! For more information on the model please refer to the model section below. **How to use the notebook**:
  - Just run the cells one after another and at the end you will have a fully trained model ready to test with real life data
  - The Notebook uses the python script "Cleaning_pipeline.py" to clean the training data. We then use this data in the notebook to train the model. Feel free to play with this script as well, as here is where all the "magic" happens. **Maybe you have a great idea to improve the accuracy of the model! Try and let us know how it went!**
  - A predictor for NaN values of Customers is saved as a Pickle File under the name "customer_pred_model.sav". For more information regarding this file please refer to the Model section below

## Before you start!

Before you start, you need to set up your environment and unzip the data. Make sure you have conda installed.

### Setup the environment
please run the following command in your favorite terminal. The Environment is called "MiniComp"

```bash

conda env create -f environment.yml

```


### Get the data

please run the following command in your terminal. Be aware this is not the same as unzipping as the data was modified for this challenge
```bash

python data.py --test 1
```

**Now you are good to go and use the notebooks!**

## The Dataset

The dataset is made of two csvs:

```
#  store.csv
['Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

#  train.csv
['Date', 'Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo','StateHoliday', 'SchoolHoliday']
```

Feature information from Kaggle:

```
Id - an Id that represents a (Store, Date) duple within the test set

Store - a unique Id for each store

Sales - the turnover for any given day (this is what you are predicting)

Customers - the number of customers on a given day

Open - an indicator for whether the store was open: 0 = closed, 1 = open

StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None

SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools

StoreType - differentiates between 4 different store models: a, b, c, d

Assortment - describes an assortment level: a = basic, b = extra, c = extended

CompetitionDistance - distance in meters to the nearest competitor store

CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened

Promo - indicates whether a store is running a promo on that day

Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating

Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
```

The test period is from 2013-01-01 to 2014-07-31 - the test dataset is the same format as `train.csv`. One thing to note here is, even though the test set contains customers, we should consider this feature to be unknown!!


## Predictive accuracy

The goal is to predict the `Sales` of a given store on a given day. The metric used to meassure accuracy is given by the root mean square percentage error (RMSPE):

![](./assets/rmspe.png)

```python
def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])
```

Zero sales days are ignored in scoring.


## Model

### The Best Model: Gradient Boost Tree
The model we use is a Gradient Boost Tree from XGBoost. The Hyper-parameters we used are:
- Max depth for each tree: 3
- Learning rate: 0.5
- Number of iterations: 15,000
- Regularization (Lambda): 4
- Subsample: 0.8

We tested a lot of different values for the above mentioned parameters, and found these to be the best mix. 

### Other Models:
We also tried other popular algorithmic models, however none of them could achieve the same performance as the XGBoost. The next bests were: Random Forest, Tree, Average Sales per Store and Linear Regression. Feel Free to try them by yourself.

### Feature Engineering
To increase the prediction power of our models we invested a lot of effort engineering new features. Please go to the python scripts "Cleaning_pipeline.py" or "Rossman_cleaning_pipeline.py" if you are interested in the implementation of these features.

Furthermore, we use a predictor for the number of customers based on the sales, store and day of week, since there were many NaN values for the number of Customers when there were positive sales. This was the case for around 15,000 entries. We train this predictor also using XGBoost and saved it as a Pickle file "customer_pred_model.sav". We did not saved its implementation but feel free to try it by yourself.
