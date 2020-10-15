## Rossman Kaggle Mini-Competition

This mini competition is adapted from the Kaggle Rossman challenge and was part of the Data Science Retreat Program 2020.
The Task is to predict future sales of given stores, using the data from two csv files. 
One file contains the past performances of stores on given dates and the other contains information about the particular stores such as distance to competition.



To run the complete pipeline, which includes data cleaning, model training and ends in the model prediction run the Rossman_sales_pred_notebook.ipy. 
Make sure to run: python data.py --test 1 first to create the data.



## Setup

```bash

#  Before engaging in the notebook run this command to create the data from the zip-file (Be aware this is not the same as unzipping as the data was modified for this challenge)
python data.py --test 1
```

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

The task is to predict the `Sales` of a given store on a given day.

Submissions are evaluated on the root mean square percentage error (RMSPE):

![](./assets/rmspe.png)

```python
def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])
```

Zero sales days are ignored in scoring.


## Notebook explanation





## Reproducibility

The entire model should be completely reproducible - to score this the teacher will clone your repository and follow the instructions as per the readme.  All teams start out with a score of 10.  One point is deducted for each step not included in the repo.

## Advice

Commit early and often

Notebooks don't merge easily!

Visualize early

Look at the predictions your model is getting wrong - can you engineer a feature for those samples?

Models
- baseline (average sales per store from in training data)
- random forest
- XGBoost

Use your DSR instructor(s)
- you are not alone - they are here to help with both bugs and data science advice
- git issues, structuring the data on disk, models to try, notebook problems and conda problems are all things we have seen before
