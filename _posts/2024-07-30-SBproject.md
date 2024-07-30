---
title: "Starbucks Capstone Challenge"
date: 2024-07-30
---

![alt text](https://github.com/fatinshariff/skills-github-pages/_images/output_110_0.png?raw=true)

# Starbucks Capstone Challenge

### Introduction

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. 

Not all users receive the same offer, and that is the challenge to solve with this data set.

The task is to combine transaction, demographic and offer data to determine **which demographic groups respond best to which offer type**. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, we can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

Transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase is given. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer. 

Keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.

### Example

To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.

However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.

### Cleaning

This makes data cleaning especially important and tricky.

We also need to take into account that some demographic groups will make purchases even if they don't receive an offer. From a business perspective, if a customer is going to make a 10 dollar purchase without an offer anyway, we wouldn't want to send a buy 10 dollars get 2 dollars off offer. We'll want to try to assess what a certain demographic group will buy when not receiving any offers.


# Data Sets

The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record


# Data Cleaning 

## Datasets Loading


```python
#import all neccessary libraries

import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
%matplotlib inline

#libraries for ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from time import time

# read in the json files
df1 = pd.read_json('data/portfolio.json', orient='records', lines=True)
df2 = pd.read_json('data/profile.json', orient='records', lines=True)
df3 = pd.read_json('data/transcript.json', orient='records', lines=True)
```

## Datasets Cleaning

### Portfolio
There is no missing values or duplicates in this file. 
* rename column id to offer id
* change id to numerical offer id
* create a dummy column for offer_type (informational,discount,bogo)
* create dummy columns for channels


```python
portfolio = df1.copy()
portfolio.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reward</th>
      <th>channels</th>
      <th>difficulty</th>
      <th>duration</th>
      <th>offer_type</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>[email, mobile, social]</td>
      <td>10</td>
      <td>7</td>
      <td>bogo</td>
      <td>ae264e3637204a6fb9bb56bc8210ddfd</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>[web, email, mobile, social]</td>
      <td>10</td>
      <td>5</td>
      <td>bogo</td>
      <td>4d5c57ea9a6940dd891ad53e9dbe8da0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>[web, email, mobile]</td>
      <td>0</td>
      <td>4</td>
      <td>informational</td>
      <td>3f207df678b143eea3cee63160fa8bed</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>[web, email, mobile]</td>
      <td>5</td>
      <td>7</td>
      <td>bogo</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>[web, email]</td>
      <td>20</td>
      <td>10</td>
      <td>discount</td>
      <td>0b1e1539f2cc45b7b9fa7c272da2e1d7</td>
    </tr>
  </tbody>
</table>
</div>




```python
#rename column id to offer_id
portfolio = portfolio.rename(columns={"id":"offer_id"})
```


```python
#encode offer ids in portfolio dataframe from string format to integer
offer_ids = portfolio['offer_id'].unique()
offer_ids_dict = pd.Series(offer_ids).to_dict()
offer_ids_dict = dict([(value,key) for key,value in offer_ids_dict.items()])
```


```python
offer_ids_dict
```




    {'ae264e3637204a6fb9bb56bc8210ddfd': 0,
     '4d5c57ea9a6940dd891ad53e9dbe8da0': 1,
     '3f207df678b143eea3cee63160fa8bed': 2,
     '9b98b8c7a33c4b65b9aebfe6a799e6d9': 3,
     '0b1e1539f2cc45b7b9fa7c272da2e1d7': 4,
     '2298d6c36e964ae4a3e7e9706d1fb8c2': 5,
     'fafdcd668e3743c1bb461111dcafc2a4': 6,
     '5a8bc65990b245e5a138643cd4eb9837': 7,
     'f19421c1d4aa40978ebb69ca19b0e20d': 8,
     '2906b810c7d4411798c6938adc9daaa5': 9}




```python
#map offer id in portfolio to the encoded offer id
portfolio['offer_id'] = portfolio['offer_id'].map(offer_ids_dict)
```


```python
#get the dummy columns for the offer_type
dummy_offertype = pd.get_dummies(portfolio.offer_type, prefix=None, dtype=int)

#merge the dummy columns with the portfolio df
portfolio = pd.concat([portfolio,dummy_offertype],axis=1)

#drop the offer_type column
# portfolio.drop(['offer_type'],axis =1, inplace= True)
```


```python
#get dummies for column 'channels'
dummy = pd.get_dummies(portfolio.channels.apply(pd.Series).stack()).groupby(level=0).sum()

#merge the dummy with the portfolio df
portfolio = pd.concat([portfolio,dummy],axis=1)

#drop the channels column
portfolio.drop(['channels'],axis =1, inplace= True)
```


```python
portfolio.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reward</th>
      <th>difficulty</th>
      <th>duration</th>
      <th>offer_type</th>
      <th>offer_id</th>
      <th>bogo</th>
      <th>discount</th>
      <th>informational</th>
      <th>email</th>
      <th>mobile</th>
      <th>social</th>
      <th>web</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>10</td>
      <td>7</td>
      <td>bogo</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>10</td>
      <td>5</td>
      <td>bogo</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>informational</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>5</td>
      <td>7</td>
      <td>bogo</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>20</td>
      <td>10</td>
      <td>discount</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Profile


```python
profile = df2.copy()
profile.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>id</th>
      <th>became_member_on</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>118</td>
      <td>68be06ca386d4c31939f3a4f0e3dd783</td>
      <td>20170212</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>55</td>
      <td>0610b486422d4921ae7d2bf64640c50b</td>
      <td>20170715</td>
      <td>112000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>None</td>
      <td>118</td>
      <td>38fe809add3b4fcf9315a9694bb96ff5</td>
      <td>20180712</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F</td>
      <td>75</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>20170509</td>
      <td>100000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>118</td>
      <td>a03223e636434f42ac4c3df47e8bac43</td>
      <td>20170804</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
profile.shape
```




    (17000, 5)




```python
profile.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>became_member_on</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>17000.000000</td>
      <td>1.700000e+04</td>
      <td>14825.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>62.531412</td>
      <td>2.016703e+07</td>
      <td>65404.991568</td>
    </tr>
    <tr>
      <th>std</th>
      <td>26.738580</td>
      <td>1.167750e+04</td>
      <td>21598.299410</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>2.013073e+07</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>45.000000</td>
      <td>2.016053e+07</td>
      <td>49000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>58.000000</td>
      <td>2.017080e+07</td>
      <td>64000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>73.000000</td>
      <td>2.017123e+07</td>
      <td>80000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>118.000000</td>
      <td>2.018073e+07</td>
      <td>120000.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
profile.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 17000 entries, 0 to 16999
    Data columns (total 5 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   gender            14825 non-null  object 
     1   age               17000 non-null  int64  
     2   id                17000 non-null  object 
     3   became_member_on  17000 non-null  int64  
     4   income            14825 non-null  float64
    dtypes: float64(1), int64(2), object(2)
    memory usage: 664.2+ KB



```python
profile[profile.age == 118].sort_values('income', ascending = False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>id</th>
      <th>became_member_on</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>118</td>
      <td>68be06ca386d4c31939f3a4f0e3dd783</td>
      <td>20170212</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>None</td>
      <td>118</td>
      <td>38fe809add3b4fcf9315a9694bb96ff5</td>
      <td>20180712</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>118</td>
      <td>a03223e636434f42ac4c3df47e8bac43</td>
      <td>20170804</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>None</td>
      <td>118</td>
      <td>8ec6ce2a7e7949b1bf142def7d0e0586</td>
      <td>20170925</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>None</td>
      <td>118</td>
      <td>68617ca6246f4fbc85e91a2a49552598</td>
      <td>20171002</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16980</th>
      <td>None</td>
      <td>118</td>
      <td>5c686d09ca4d475a8f750f2ba07e0440</td>
      <td>20160901</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16982</th>
      <td>None</td>
      <td>118</td>
      <td>d9ca82f550ac4ee58b6299cf1e5c824a</td>
      <td>20160415</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16989</th>
      <td>None</td>
      <td>118</td>
      <td>ca45ee1883624304bac1e4c8a114f045</td>
      <td>20180305</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16991</th>
      <td>None</td>
      <td>118</td>
      <td>a9a20fa8b5504360beb4e7c8712f8306</td>
      <td>20160116</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16994</th>
      <td>None</td>
      <td>118</td>
      <td>c02b10e8752c4d8e9b73f918558531f7</td>
      <td>20151211</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2175 rows × 5 columns</p>
</div>




```python
#checking if it has any value for gender and income
profile[profile.age == 118].gender.notna().sum() , profile[profile.age == 118].income.notna().sum()

```




    (0, 0)



There is a high count for the outliers age (age = 118) which is more than 2k. This could be when customer didn't enter their age in the form, it will automatically set to a default date. 

We have to take care of missing values if we want to do modelling later. There are three options that we have; get rid of the corresponding districts, get rid of the whole attributes or replace the values to some value. So, in this case we are going to remove this data because it's representing false age and even the income and gender data are missing.


```python
profile = profile[profile['age']<118].reset_index(drop=True)
```


```python
profile.age.sort_values()
```




    3469      18
    5910      18
    5151      18
    1964      18
    645       18
            ... 
    13787    101
    3548     101
    12969    101
    14707    101
    1340     101
    Name: age, Length: 14825, dtype: int64




```python
profile.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14825 entries, 0 to 14824
    Data columns (total 5 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   gender            14825 non-null  object 
     1   age               14825 non-null  int64  
     2   id                14825 non-null  object 
     3   became_member_on  14825 non-null  int64  
     4   income            14825 non-null  float64
    dtypes: float64(1), int64(2), object(2)
    memory usage: 579.2+ KB



```python
profile.duplicated().sum()
```




    0



From the information above, we see that there is no missing value or duplicated rows.

So the next cleaning process for this dataset will be:

- change id to customer_id
- change customer_id to encoded id
- change datatype for column became_member_on to date


```python
#rename id to customer_id
profile = profile.rename(columns = {'id':'customer_id'})

customer_ids = pd.unique(profile['customer_id'])

#encode customer ids which is in string format to integers
customer_ids_dict = pd.Series(customer_ids).to_dict()

#swap value and key
customer_ids_dict = dict([(value, key) for key, value in customer_ids_dict.items()]) 

#map the new encoded customer id to the old customer id in profile table
profile['customer_id'] = profile['customer_id'].map(customer_ids_dict)
```


```python
#change datatype column 'became_member_on' to datetime
profile['became_member_on'] = pd.to_datetime(profile['became_member_on'],format='%Y%m%d')
```


```python
profile.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>customer_id</th>
      <th>became_member_on</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>F</td>
      <td>55</td>
      <td>0</td>
      <td>2017-07-15</td>
      <td>112000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>75</td>
      <td>1</td>
      <td>2017-05-09</td>
      <td>100000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>68</td>
      <td>2</td>
      <td>2018-04-26</td>
      <td>70000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>65</td>
      <td>3</td>
      <td>2018-02-09</td>
      <td>53000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>58</td>
      <td>4</td>
      <td>2017-11-11</td>
      <td>51000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
profile.dtypes
```




    gender                      object
    age                          int64
    customer_id                  int64
    became_member_on    datetime64[ns]
    income                     float64
    dtype: object



### Transcript


```python
transcript = df3.copy()
transcript.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person</th>
      <th>event</th>
      <th>value</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer received</td>
      <td>{'offer id': '9b98b8c7a33c4b65b9aebfe6a799e6d9'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a03223e636434f42ac4c3df47e8bac43</td>
      <td>offer received</td>
      <td>{'offer id': '0b1e1539f2cc45b7b9fa7c272da2e1d7'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e2127556f4f64592b11af22de27a7932</td>
      <td>offer received</td>
      <td>{'offer id': '2906b810c7d4411798c6938adc9daaa5'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8ec6ce2a7e7949b1bf142def7d0e0586</td>
      <td>offer received</td>
      <td>{'offer id': 'fafdcd668e3743c1bb461111dcafc2a4'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>68617ca6246f4fbc85e91a2a49552598</td>
      <td>offer received</td>
      <td>{'offer id': '4d5c57ea9a6940dd891ad53e9dbe8da0'}</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
transcript.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>306534.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>366.382940</td>
    </tr>
    <tr>
      <th>std</th>
      <td>200.326314</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>186.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>408.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>528.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>714.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
transcript.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 306534 entries, 0 to 306533
    Data columns (total 4 columns):
     #   Column  Non-Null Count   Dtype 
    ---  ------  --------------   ----- 
     0   person  306534 non-null  object
     1   event   306534 non-null  object
     2   value   306534 non-null  object
     3   time    306534 non-null  int64 
    dtypes: int64(1), object(3)
    memory usage: 9.4+ MB



```python
transcript.event.unique()
```




    array(['offer received', 'offer viewed', 'transaction', 'offer completed'],
          dtype=object)




```python
transcript.event.value_counts() 
```




    event
    transaction        138953
    offer received      76277
    offer viewed        57725
    offer completed     33579
    Name: count, dtype: int64



Data cleaning to do:

- change column name from 'person' to 'customer_id'
- convert the column 'Event' into 4 different columns based on their value (offer received/offer viewed/offer completed/transaction)
- convert the column 'Values' into columns according to the value's dictionary keys
- map encoded customer ids to ids in transcript and profile dataframes



```python
#change person to 'customer_id'
transcript = transcript.rename(columns={'person':'customer_id'})
```


```python
#replace space with underscore for values in event column
transcript['event'] = transcript.event.str.replace(' ','_')
```


```python
#Convert categorical variable into dummy variables.
dummy = pd.get_dummies(transcript.event,dtype=int)

# merge back all the dummy variables
transcript = pd.concat([transcript,dummy], axis=1)

#drop the event column 
transcript = transcript.drop('event',axis= 1)
```


```python
transcript.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 306534 entries, 0 to 306533
    Data columns (total 7 columns):
     #   Column           Non-Null Count   Dtype 
    ---  ------           --------------   ----- 
     0   customer_id      306534 non-null  object
     1   value            306534 non-null  object
     2   time             306534 non-null  int64 
     3   offer_completed  306534 non-null  int64 
     4   offer_received   306534 non-null  int64 
     5   offer_viewed     306534 non-null  int64 
     6   transaction      306534 non-null  int64 
    dtypes: int64(5), object(2)
    memory usage: 16.4+ MB

# transcript[transcript.offer_completed ==1]
transcript[transcript.transaction==1]

```python
dummy_val = transcript['value'].apply(pd.Series)
```


```python
dummy_val.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 306534 entries, 0 to 306533
    Data columns (total 4 columns):
     #   Column    Non-Null Count   Dtype  
    ---  ------    --------------   -----  
     0   offer id  134002 non-null  object 
     1   amount    138953 non-null  float64
     2   offer_id  33579 non-null   object 
     3   reward    33579 non-null   float64
    dtypes: float64(2), object(2)
    memory usage: 9.4+ MB

dummy_val[dummy_val["amount"].notna()]
Rename the 'offer id' to 'offer_id' and combine with the existing 'offer_id'. We can drop the reward column as it is already captured in portfolio dataframe.
dummy_val[dummy_val.offer_id.notna()]

```python
dummy_val.offer_id.fillna(dummy_val['offer id'],inplace = True)
```

    /var/folders/3t/k2d3c7k50ybfm83nhrzl90gh0000gn/T/ipykernel_50132/3872812856.py:1: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      dummy_val.offer_id.fillna(dummy_val['offer id'],inplace = True)



```python
dummy_val = dummy_val.drop(['offer id','reward'],axis = 1)
```


```python
dummy_val.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount</th>
      <th>offer_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>0b1e1539f2cc45b7b9fa7c272da2e1d7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>2906b810c7d4411798c6938adc9daaa5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>fafdcd668e3743c1bb461111dcafc2a4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>4d5c57ea9a6940dd891ad53e9dbe8da0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#combined the dummy_val to the transcript df
transcript = pd.concat([transcript,dummy_val], axis = 1)

#drop the value column
transcript = transcript.drop('value', axis = 1)
```


```python
#map encoded customer ids to ids in transcrpt dataframes
transcript['customer_id'] = transcript['customer_id'].map(customer_ids_dict)
```


```python
#map offer id in transcript to the encoded offer id
transcript['offer_id'] = transcript['offer_id'].map(offer_ids_dict)
```


```python
transcript.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 306534 entries, 0 to 306533
    Data columns (total 8 columns):
     #   Column           Non-Null Count   Dtype  
    ---  ------           --------------   -----  
     0   customer_id      272762 non-null  float64
     1   time             306534 non-null  int64  
     2   offer_completed  306534 non-null  int64  
     3   offer_received   306534 non-null  int64  
     4   offer_viewed     306534 non-null  int64  
     5   transaction      306534 non-null  int64  
     6   amount           138953 non-null  float64
     7   offer_id         167581 non-null  float64
    dtypes: float64(3), int64(5)
    memory usage: 18.7 MB



```python
#drop all rows contain NA because they are the customer with age~118 that we removed earlier in profile table

transcript.dropna(axis = 0, subset = ['customer_id'], inplace = True)
```


```python
transcript.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 272762 entries, 0 to 306532
    Data columns (total 8 columns):
     #   Column           Non-Null Count   Dtype  
    ---  ------           --------------   -----  
     0   customer_id      272762 non-null  float64
     1   time             272762 non-null  int64  
     2   offer_completed  272762 non-null  int64  
     3   offer_received   272762 non-null  int64  
     4   offer_viewed     272762 non-null  int64  
     5   transaction      272762 non-null  int64  
     6   amount           123957 non-null  float64
     7   offer_id         148805 non-null  float64
    dtypes: float64(3), int64(5)
    memory usage: 18.7 MB


### Combining Dataset


```python
df = pd.merge(profile,transcript, how = 'outer', on = 'customer_id')
df = pd.merge(df,portfolio, how = 'outer', on = 'offer_id')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>customer_id</th>
      <th>became_member_on</th>
      <th>income</th>
      <th>time</th>
      <th>offer_completed</th>
      <th>offer_received</th>
      <th>offer_viewed</th>
      <th>transaction</th>
      <th>...</th>
      <th>difficulty</th>
      <th>duration</th>
      <th>offer_type</th>
      <th>bogo</th>
      <th>discount</th>
      <th>informational</th>
      <th>email</th>
      <th>mobile</th>
      <th>social</th>
      <th>web</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>F</td>
      <td>75</td>
      <td>1</td>
      <td>2017-05-09</td>
      <td>100000.0</td>
      <td>408</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>75</td>
      <td>1</td>
      <td>2017-05-09</td>
      <td>100000.0</td>
      <td>408</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F</td>
      <td>75</td>
      <td>1</td>
      <td>2017-05-09</td>
      <td>100000.0</td>
      <td>510</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F</td>
      <td>61</td>
      <td>5</td>
      <td>2017-09-11</td>
      <td>57000.0</td>
      <td>408</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>F</td>
      <td>61</td>
      <td>5</td>
      <td>2017-09-11</td>
      <td>57000.0</td>
      <td>426</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



# Data Visualisation

## Offers

In the first part, let's explore the offer types. Let find out the business funnel for the offers and what criteria does the offer has for the higher conversion. 


```python
df.groupby(['offer_id'])[['offer_received','offer_viewed','offer_completed']].sum().plot.bar()
plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.);
```


    
![png](output_60_0.png)
    


It's hard to deduce from the chart above but we can see some of the have 0 completion. Let's plot them base of the offer type instead of offer individually.


```python
offer_type_funnel = df.groupby('offer_type')[['offer_received','offer_viewed','offer_completed']].sum()
offer_type_funnel
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>offer_received</th>
      <th>offer_viewed</th>
      <th>offer_completed</th>
    </tr>
    <tr>
      <th>offer_type</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bogo</th>
      <td>26537</td>
      <td>22039</td>
      <td>15258</td>
    </tr>
    <tr>
      <th>discount</th>
      <td>26664</td>
      <td>18461</td>
      <td>17186</td>
    </tr>
    <tr>
      <th>informational</th>
      <td>13300</td>
      <td>9360</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Since this is the sum value for all the offers base on the offer type, we are going to average them base on the offer_type count and plot the number in a bar graph.


```python
#divider is the counts of each offer type
divider = portfolio.offer_type.value_counts().values

#average received,viewed,and completed value for each offer base on their type
(offer_type_funnel.T/divider).T.plot.bar()
plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.)
plt.title("Average Event's Count for each Offer Type");
```


    
![png](output_64_0.png)
    


To make a fair comparison, let's normalize the above value and replot them.


```python
a = pd.Series((offer_type_funnel.T/divider).iloc[0]/(offer_type_funnel.T/divider).iloc[0], name = 'Received')
b = pd.Series((offer_type_funnel.T/divider).iloc[1]/(offer_type_funnel.T/divider).iloc[0], name = 'Viewed')
c = pd.Series((offer_type_funnel.T/divider).iloc[2]/(offer_type_funnel.T/divider).iloc[1], name = 'Completed')
```


```python
ab = pd.merge(a,b, on = 'offer_type')
pd.merge(ab,c, on = 'offer_type').plot.bar()
plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.)
plt.title("Percentage Event for each Offer Type");
```


    
![png](output_67_0.png)
    


From the chart above, we can see that customer seems to be most interested in BOGO offer, hence it has the highest view but it doesn't mean that it has the highest conversion. The discout type offer seems to have the highest conversion which means most of them that viewed the offer has high chance of completing the offer. Meanwhile, the informational offer seems to have no conversion at all, maybe because customer realized they are not gaining anything here.

How about the difficulty of the offer? Does it show any trend on completed offer counts? Let's explore that below.


```python
offer_type_funnel = df.groupby('difficulty')[['offer_received','offer_viewed','offer_completed']].sum()
offer_type_funnel
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>offer_received</th>
      <th>offer_viewed</th>
      <th>offer_completed</th>
    </tr>
    <tr>
      <th>difficulty</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>13300</td>
      <td>9360</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>13261</td>
      <td>9809</td>
      <td>8291</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>6655</td>
      <td>6379</td>
      <td>4886</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>26559</td>
      <td>22097</td>
      <td>15881</td>
    </tr>
    <tr>
      <th>20.0</th>
      <td>6726</td>
      <td>2215</td>
      <td>3386</td>
    </tr>
  </tbody>
</table>
</div>




```python
#divider is the counts of each offer type
divider = portfolio.difficulty.value_counts().sort_index().values
#average received,viewed,and completed value for each offer base on their difficulty
(offer_type_funnel.T/divider).T.plot.bar();
plt.title("Customer Counts per Difficulty")
plt.ylabel('Counts')
plt.xlabel('Difficulty')
plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.)

a = pd.Series((offer_type_funnel.T/divider).iloc[0]/(offer_type_funnel.T/divider).iloc[0], name = 'Received')
b = pd.Series((offer_type_funnel.T/divider).iloc[1]/(offer_type_funnel.T/divider).iloc[0], name = 'Viewed')
c = pd.Series((offer_type_funnel.T/divider).iloc[2]/(offer_type_funnel.T/divider).iloc[1], name = 'Completed')

ab = pd.merge(a,b, on = 'difficulty')
pd.merge(ab,c, on = 'difficulty').plot.bar()
plt.title("Customer Proportion per Difficulty")
plt.ylabel('Percentage')
plt.xlabel('Difficulty')
plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.);
```


    
![png](output_71_0.png)
    



    
![png](output_71_1.png)
    



```python
portfolio.groupby(['offer_type','difficulty']).offer_id.count()
```




    offer_type     difficulty
    bogo           5             2
                   10            2
    discount       7             1
                   10            2
                   20            1
    informational  0             2
    Name: offer_id, dtype: int64



0 difficulty is 100 percent coming from informational offer. Most people viewed the  difficulty between 5-10 with 7 having the highest count for both views and completions.
The completed offer for the most difficult offer has more than 100% completions from the total viewed. This could obviously resulted by the inclusion of the demographic who doesn't even received the offer but still make the purchase anyway (which was explain earlier in the introduction).

More ideas for conclusion later:

 - what is this informational offer actually. because the 0 completed offer kind of doesn'y make sense. I assume that if they release new product or seasonal drinks, they sent the information about this to their customers. In reality there must have been people buying the new drinks. So in this case the 0 doesn't make sense.
 
 - If that's the case, that means that this informational offer are giving them negative return of investment (ROI) and they should stop with the offer.

## Distributions of Customer's Age,Gender and Income


Now, let's explore the customer's data. Let's find out about the age, gender and are there any correlation between the customer and the offer.


```python
profile.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>customer_id</th>
      <th>became_member_on</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>F</td>
      <td>55</td>
      <td>0</td>
      <td>2017-07-15</td>
      <td>112000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>75</td>
      <td>1</td>
      <td>2017-05-09</td>
      <td>100000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>68</td>
      <td>2</td>
      <td>2018-04-26</td>
      <td>70000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>65</td>
      <td>3</td>
      <td>2018-02-09</td>
      <td>53000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>58</td>
      <td>4</td>
      <td>2017-11-11</td>
      <td>51000.0</td>
    </tr>
  </tbody>
</table>
</div>



Let's look at the age distribution after we have remove the maximum age of 118 in the earlier cleaning part.


```python
# profile = profile[profile.age < profile.age.max()]
plt.hist(profile.age)
plt.grid()
plt.title("Customer's Age Distribution")
plt.xlabel("Age");
```


    
![png](output_77_0.png)
    


They are normally distributed with median around 55 of age and slightly skewed to the right.


```python
plt.hist(profile.income)
plt.grid()
plt.title("Customer's Income Distribution")
plt.xlabel("Income");
```


    
![png](output_79_0.png)
    



```python
profile[['age','income']].hist(bins =40,figsize =(30,15))
plt.show;
```


    
![png](output_80_0.png)
    



```python
#cust age and income distribution

profile.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>customer_id</th>
      <th>became_member_on</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14825.000000</td>
      <td>14825.000000</td>
      <td>14825</td>
      <td>14825.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>54.393524</td>
      <td>7412.000000</td>
      <td>2017-02-18 12:30:15.419898880</td>
      <td>65404.991568</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>2013-07-29 00:00:00</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>42.000000</td>
      <td>3706.000000</td>
      <td>2016-05-20 00:00:00</td>
      <td>49000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>55.000000</td>
      <td>7412.000000</td>
      <td>2017-08-02 00:00:00</td>
      <td>64000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>66.000000</td>
      <td>11118.000000</td>
      <td>2017-12-30 00:00:00</td>
      <td>80000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>101.000000</td>
      <td>14824.000000</td>
      <td>2018-07-26 00:00:00</td>
      <td>120000.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>17.383705</td>
      <td>4279.753206</td>
      <td>NaN</td>
      <td>21598.299410</td>
    </tr>
  </tbody>
</table>
</div>




```python
#scatter plot to see relationship between age and income
plt.scatter(profile.age, profile.income, alpha = 0.05)
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Income and Age Scatter Plot');
```


    
![png](output_82_0.png)
    


The above plot some how show the data are being capped multiple times for both age and income.
The income is capped at around 75k in the first level, then at 100k and 120k for the subsequent levels. Same goes to the age where we can see the age start at 18 years old, then there is another minimum cap that start around mid 30s and another one around 50 years old. 


```python
#just checking if the data is actually capped based on the time they became a member

profile.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14825 entries, 0 to 14824
    Data columns (total 5 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   gender            14825 non-null  object        
     1   age               14825 non-null  int64         
     2   customer_id       14825 non-null  int64         
     3   became_member_on  14825 non-null  datetime64[ns]
     4   income            14825 non-null  float64       
    dtypes: datetime64[ns](1), float64(1), int64(2), object(1)
    memory usage: 579.2+ KB



```python
plt.plot(profile.groupby('became_member_on')[['income']].max())
plt.xlabel("Membership Date")
plt.title("Maximum Income of Customer");
```


    
![png](output_85_0.png)
    



```python
plt.plot(profile.groupby('became_member_on')[['income']].min())
plt.xlabel("Membership Date")
plt.title("Minimum Income of Customer");
```


    
![png](output_86_0.png)
    


It's interesting how the maximum income for who became Starbuck's member before midyear 2015 was at 100k and right after that period, the value was at 120k. 

Meanwhile the minimum income is showing an obvious different trend before and after mid 2017. This somehow conclude that there are three different timeframe data, mid 2013 to mid 2015, mid 2015 to mid 2017 and lastly after mid 2017. This could be the reason behind the capped data in the scatter plot on the income and age above.  

Now let's explore how gender in the dataset is divided.


```python
profile.gender.value_counts()
```




    gender
    M    8484
    F    6129
    O     212
    Name: count, dtype: int64




```python
# profile.gender.value_counts(normalize = True).plot(kind = 'bar')
gender = ['Male', 'Female', 'Other']
gender_counts = profile.gender.value_counts().values

# Create a pie chart of the number of customers for each gender
plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender, autopct="%1.1f%%")
plt.title("Pie Chart of Customer Gender Distribution")
plt.show()
```


    
![png](output_90_0.png)
    


Majority of the customer are male with value of more than 50% of them. 

Is there a pattern where woman are more prone to complete the offer compare to man? Let's find about that.

## Relationships between Customer's Feature and the Offers


We realized earlier that the majority of our customer are male but does it apply the same for those who completed the offer? Let's find out on that.


```python
#create a customer table that holds information on their response towards the offer
cust = df.groupby('customer_id')[['transaction','offer_received','offer_viewed','offer_completed','amount']].sum()
#sort the table by the highest value of transaction
cust = cust.sort_values(by = ['transaction'], ascending = False)
#add gender info
cust['gender'] = df.groupby('customer_id')['gender'].max()
```


```python
#average of offer completed by each gender
gender_completed = cust.groupby('gender').offer_completed.mean()

```


```python
#setting up values for x and y axis
gender = ['Female', 'Male', 'Other']
y_val = gender_completed.values

# Create a bar chart of the number of customers for each gender
plt.figure(figsize=(6, 6))
plt.bar(gender, y_val)
plt.title("Average Offer Completed by Gender")
plt.show()
```


    
![png](output_96_0.png)
    


From the about result, we see that our male customer is in contrary contribute to least completed offer based on average per person. This is an interesting find where women and other gender tend to complete the offer most. 



Let's move on to the next question, how much transactions does the customer make and what's their spending pattern? 


```python
# adding more columns to the new cust table
cust['transaction_no_offer'] = cust.transaction - cust.offer_completed
cust['income'] = df.groupby('customer_id')['income'].max()
cust['age'] = df.groupby('customer_id')['age'].max()

```


```python
cust.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transaction</th>
      <th>offer_received</th>
      <th>offer_viewed</th>
      <th>offer_completed</th>
      <th>amount</th>
      <th>gender</th>
      <th>transaction_no_offer</th>
      <th>income</th>
      <th>age</th>
    </tr>
    <tr>
      <th>customer_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8936</th>
      <td>36</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>173.41</td>
      <td>M</td>
      <td>32</td>
      <td>34000.0</td>
      <td>76</td>
    </tr>
    <tr>
      <th>5554</th>
      <td>36</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>76.46</td>
      <td>M</td>
      <td>33</td>
      <td>48000.0</td>
      <td>63</td>
    </tr>
    <tr>
      <th>2699</th>
      <td>35</td>
      <td>6</td>
      <td>5</td>
      <td>5</td>
      <td>90.23</td>
      <td>M</td>
      <td>30</td>
      <td>33000.0</td>
      <td>27</td>
    </tr>
    <tr>
      <th>5821</th>
      <td>32</td>
      <td>6</td>
      <td>5</td>
      <td>5</td>
      <td>103.66</td>
      <td>M</td>
      <td>27</td>
      <td>55000.0</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1277</th>
      <td>32</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>133.02</td>
      <td>M</td>
      <td>28</td>
      <td>64000.0</td>
      <td>37</td>
    </tr>
    <tr>
      <th>3418</th>
      <td>31</td>
      <td>6</td>
      <td>4</td>
      <td>5</td>
      <td>461.09</td>
      <td>F</td>
      <td>26</td>
      <td>33000.0</td>
      <td>62</td>
    </tr>
    <tr>
      <th>6713</th>
      <td>30</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>220.22</td>
      <td>M</td>
      <td>26</td>
      <td>31000.0</td>
      <td>61</td>
    </tr>
    <tr>
      <th>5214</th>
      <td>30</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>260.69</td>
      <td>M</td>
      <td>24</td>
      <td>37000.0</td>
      <td>53</td>
    </tr>
    <tr>
      <th>10117</th>
      <td>30</td>
      <td>6</td>
      <td>6</td>
      <td>1</td>
      <td>83.04</td>
      <td>M</td>
      <td>29</td>
      <td>36000.0</td>
      <td>47</td>
    </tr>
    <tr>
      <th>5649</th>
      <td>30</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>717.21</td>
      <td>M</td>
      <td>25</td>
      <td>49000.0</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>




```python
cust.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transaction</th>
      <th>offer_received</th>
      <th>offer_viewed</th>
      <th>offer_completed</th>
      <th>amount</th>
      <th>gender</th>
      <th>transaction_no_offer</th>
      <th>income</th>
      <th>age</th>
    </tr>
    <tr>
      <th>customer_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8013</th>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>F</td>
      <td>0</td>
      <td>70000.0</td>
      <td>52</td>
    </tr>
    <tr>
      <th>968</th>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>F</td>
      <td>0</td>
      <td>88000.0</td>
      <td>64</td>
    </tr>
    <tr>
      <th>11498</th>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>M</td>
      <td>0</td>
      <td>52000.0</td>
      <td>61</td>
    </tr>
    <tr>
      <th>4076</th>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>M</td>
      <td>0</td>
      <td>77000.0</td>
      <td>51</td>
    </tr>
    <tr>
      <th>8934</th>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>M</td>
      <td>0</td>
      <td>85000.0</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>



Since we cant track how much customer spend for each completed offer, what we can do here is to see the percentage of the offer transaction from the total transactions.


```python
#find the % of offer to total transaction.
cust['perc_complete'] = round(cust.offer_completed/cust.transaction,2)

```


```python
cust.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transaction</th>
      <th>offer_received</th>
      <th>offer_viewed</th>
      <th>offer_completed</th>
      <th>amount</th>
      <th>gender</th>
      <th>transaction_no_offer</th>
      <th>income</th>
      <th>age</th>
      <th>perc_complete</th>
    </tr>
    <tr>
      <th>customer_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8936</th>
      <td>36</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>173.41</td>
      <td>M</td>
      <td>32</td>
      <td>34000.0</td>
      <td>76</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>5554</th>
      <td>36</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>76.46</td>
      <td>M</td>
      <td>33</td>
      <td>48000.0</td>
      <td>63</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>2699</th>
      <td>35</td>
      <td>6</td>
      <td>5</td>
      <td>5</td>
      <td>90.23</td>
      <td>M</td>
      <td>30</td>
      <td>33000.0</td>
      <td>27</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>5821</th>
      <td>32</td>
      <td>6</td>
      <td>5</td>
      <td>5</td>
      <td>103.66</td>
      <td>M</td>
      <td>27</td>
      <td>55000.0</td>
      <td>59</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>1277</th>
      <td>32</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>133.02</td>
      <td>M</td>
      <td>28</td>
      <td>64000.0</td>
      <td>37</td>
      <td>0.12</td>
    </tr>
  </tbody>
</table>
</div>




```python
#encode the column 'gender' in the string format to integer
gender_dict = {'O': 0, 'M': 1, 'F': 2}
cust['gender'] = cust['gender'].map(gender_dict)
```

Now let's look at how much each attribute correlate with the completed offer.


```python
corr_matrix = cust.corr()
corr_matrix['offer_completed'].sort_values(ascending = False)
```




    offer_completed         1.000000
    perc_complete           0.539021
    amount                  0.537203
    transaction             0.417381
    offer_viewed            0.398986
    offer_received          0.333998
    income                  0.257836
    gender                  0.166791
    transaction_no_offer    0.122426
    age                     0.114036
    Name: offer_completed, dtype: float64




```python
corr_matrix = cust.corr()
corr_matrix['offer_completed'].sort_values(ascending = False)
```




    offer_completed         1.000000
    perc_complete           0.539021
    amount                  0.537203
    transaction             0.417381
    offer_viewed            0.398986
    offer_received          0.333998
    income                  0.257836
    gender                  0.166791
    transaction_no_offer    0.122426
    age                     0.114036
    Name: offer_completed, dtype: float64



The most positive correlated attribute is the perc_complete which is the amount of completed offer over the total transaction. It follows by the total amount spent by the customer. So the customer who spend more are more likely to complete the offer.

So correlation number above shows the strength of the linear relationship between the completed offers and all other numerical variables. Now let have a look at them when they are plotted in graphs.


```python
import matplotlib.pyplot as plt
import numpy as np

# # make the data
# x = cust.offer_completed
# y = cust.perc_complete

# # size and color:
# sizes = cust.amount
# # colors = {'M':'tab:blue', 'F':'tab:orange', 'O':'tab:green'}

# # plot
# fig, ax = plt.subplots()

# # ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)
# ax.scatter(x, y, s=sizes)#, c=cust['gender'].map(colors))

# # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
# #        ylim=(0, 8), yticks=np.arange(1, 8))

# plt.show()

perc_completed = cust.perc_complete
offer_com = cust.offer_completed
total_spend = cust.amount

# Scatter plot with color coding based on promo usage
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(total_spend, perc_completed, c=offer_com, cmap='viridis', edgecolors='black')

# Label axes and title
plt.xlabel('Total Amount')
plt.ylabel('Percentage Offer Completed from Total Transaction')
plt.title('Relationship between Percentage Offer Completed, Offer Completed, and Total Amount Spent')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(min(offer_com), max(offer_com)))
sm.set_array([])
plt.colorbar(sm, ax= ax, label='Offer Completed')

# Show plot
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](output_110_0.png)
    



```python
cust[cust.perc_complete > 1].sort_values(by= "perc_complete",ascending= False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transaction</th>
      <th>offer_received</th>
      <th>offer_viewed</th>
      <th>offer_completed</th>
      <th>amount</th>
      <th>gender</th>
      <th>transaction_no_offer</th>
      <th>income</th>
      <th>age</th>
      <th>perc_complete</th>
    </tr>
    <tr>
      <th>customer_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>755</th>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>4</td>
      <td>31.44</td>
      <td>2</td>
      <td>-3</td>
      <td>78000.0</td>
      <td>70</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>173</th>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>23.59</td>
      <td>2</td>
      <td>-2</td>
      <td>92000.0</td>
      <td>52</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>404</th>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>3</td>
      <td>26.70</td>
      <td>1</td>
      <td>-2</td>
      <td>98000.0</td>
      <td>50</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>12339</th>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>3</td>
      <td>11.30</td>
      <td>1</td>
      <td>-2</td>
      <td>75000.0</td>
      <td>56</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3959</th>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>3</td>
      <td>42.64</td>
      <td>1</td>
      <td>-2</td>
      <td>88000.0</td>
      <td>65</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5476</th>
      <td>5</td>
      <td>6</td>
      <td>5</td>
      <td>6</td>
      <td>125.77</td>
      <td>1</td>
      <td>-1</td>
      <td>108000.0</td>
      <td>99</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>545</th>
      <td>5</td>
      <td>6</td>
      <td>3</td>
      <td>6</td>
      <td>132.94</td>
      <td>2</td>
      <td>-1</td>
      <td>114000.0</td>
      <td>52</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>10386</th>
      <td>5</td>
      <td>6</td>
      <td>3</td>
      <td>6</td>
      <td>126.75</td>
      <td>2</td>
      <td>-1</td>
      <td>82000.0</td>
      <td>68</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>9783</th>
      <td>5</td>
      <td>6</td>
      <td>5</td>
      <td>6</td>
      <td>89.63</td>
      <td>1</td>
      <td>-1</td>
      <td>62000.0</td>
      <td>47</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>1289</th>
      <td>5</td>
      <td>6</td>
      <td>2</td>
      <td>6</td>
      <td>141.51</td>
      <td>1</td>
      <td>-1</td>
      <td>105000.0</td>
      <td>70</td>
      <td>1.2</td>
    </tr>
  </tbody>
</table>
<p>156 rows × 10 columns</p>
</div>




```python

transaction_counts = cust.transaction
offer_com = cust.offer_completed
total_spend = cust.amount


# Scatter plot with color coding based on promo usage
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(transaction_counts, total_spend, c=offer_com, cmap='viridis', edgecolors='black')

# Label axes and title
plt.xlabel('Transaction Count')
plt.ylabel('Total Amount')
plt.title('Relationship between Transaction Count, Offer Completed, and Total Amount Spent')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(min(offer_com), max(offer_com)))
sm.set_array([])
plt.colorbar(sm,ax = ax, label='Offer Completed')

# Show plot
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](output_112_0.png)
    


There is no direct correlation that we can see here. The higher transaction count doesn't mean the spending amount is also higher. The higher promo usage is in the middle area of the distribution.


```python

transaction_counts = cust.transaction
promo_usage = cust.offer_completed
income = cust.income

# Scatter plot with color coding based on promo usage
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(transaction_counts, income, c=promo_usage, cmap='viridis', edgecolors='black')

# Label axes and title
plt.xlabel('Transaction Count')
plt.ylabel('Income')
plt.title('Relationship between Transaction Count, Promo Usage, and Total Spend')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(min(promo_usage), max(promo_usage)))
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Promo Usage')

# Show plot
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](output_114_0.png)
    



```python

transaction_counts = cust.transaction
promo_usage = cust.offer_completed
age = cust.age

# Scatter plot with color coding based on promo usage
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(transaction_counts, age, c=promo_usage, cmap='viridis', edgecolors='black')

# Label axes and title
plt.xlabel('Transaction Count')
plt.ylabel('Age')
plt.title('Relationship between Transaction Count, Promo Usage, and Total Spend')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(min(promo_usage), max(promo_usage)))
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Promo Usage')

# Show plot
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](output_115_0.png)
    


This is really interesting. We see the higher promo usage, which in green to yellow color marks lies densely for transaction count roughly more than five and below 25 and the total income more than its 50th percentile to 75th percentile. People with higher income don't spend more than 20 transaction.

Lets explore this further by transforming the graph into quadrants based on income and transaction count percentiles.


```python
#Calculate Percentiles:

#calculate the 25th, 50th, and 75th percentiles for both income and transaction_counts:

income_25th = income.quantile(.25)
income_50th = income.quantile(.5)
income_75th = income.quantile(.75)

transaction_25th = np.percentile(transaction_counts, 25)
transaction_50th = np.percentile(transaction_counts, 50)
transaction_75th = np.percentile(transaction_counts, 75)
transaction_100th = np.percentile(transaction_counts, 100)


```



# Building Machine Learning Model
Build a Machine Learning model to predict response of a customer to an offer



## Data Preparation

Before we strat training our model, we need to clean our dataset first. Some on the steps to be taken are as below:
- impute or remove missing value?
- making sure all are numerical attributes and drop all non numerical
- rescale using MinMaxScaler on attributes so they range from 0 to 1
- separate the target and features columns


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 272762 entries, 0 to 272761
    Data columns (total 23 columns):
     #   Column            Non-Null Count   Dtype         
    ---  ------            --------------   -----         
     0   gender            272762 non-null  object        
     1   age               272762 non-null  int64         
     2   customer_id       272762 non-null  int64         
     3   became_member_on  272762 non-null  datetime64[ns]
     4   income            272762 non-null  float64       
     5   time              272762 non-null  int64         
     6   offer_completed   272762 non-null  int64         
     7   offer_received    272762 non-null  int64         
     8   offer_viewed      272762 non-null  int64         
     9   transaction       272762 non-null  int64         
     10  amount            123957 non-null  float64       
     11  offer_id          148805 non-null  float64       
     12  reward            148805 non-null  float64       
     13  difficulty        148805 non-null  float64       
     14  duration          148805 non-null  float64       
     15  offer_type        148805 non-null  object        
     16  bogo              148805 non-null  float64       
     17  discount          148805 non-null  float64       
     18  informational     148805 non-null  float64       
     19  email             148805 non-null  float64       
     20  mobile            148805 non-null  float64       
     21  social            148805 non-null  float64       
     22  web               148805 non-null  float64       
    dtypes: datetime64[ns](1), float64(13), int64(7), object(2)
    memory usage: 47.9+ MB


Base on the information above, we have almost have of the information missing. The model we want to train later is to see which customer is more prone to complete the offer base on which offer. Hence, if we have missing information on either of the customer or offer, we are to going to remove those rows. 


```python
df = df[df.offer_id.notna()]
```


```python
# df['gender'] = df.gender.replace(['M','F','O'],[1,2,3])
#process categorical variables
categorical = ['gender']
df = pd.get_dummies(df, columns = categorical,dtype =int)
```


```python
df = df.drop(labels=['became_member_on','amount','offer_type','customer_id','offer_id','offer_received','offer_viewed'], axis=1)
df.columns
```




    Index(['age', 'income', 'time', 'offer_completed', 'transaction', 'reward',
           'difficulty', 'duration', 'bogo', 'discount', 'informational', 'email',
           'mobile', 'social', 'web', 'gender_F', 'gender_M', 'gender_O'],
          dtype='object')




```python
df.iloc[3]
```




    age                   61.0
    income             57000.0
    time                 408.0
    offer_completed        0.0
    transaction            0.0
    reward                10.0
    difficulty            10.0
    duration               7.0
    bogo                   1.0
    discount               0.0
    informational          0.0
    email                  1.0
    mobile                 1.0
    social                 1.0
    web                    0.0
    gender_F               1.0
    gender_M               0.0
    gender_O               0.0
    Name: 3, dtype: float64




```python
#process numerical variables
#initialize a MinMaxScaler, then apply it to the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age','income','time', 'reward','difficulty', 'duration']
df[numerical] = scaler.fit_transform(df[numerical])
```


```python
target = df.offer_completed
features = df.drop('offer_completed', axis = 1)
print(" Number of total features: {} ".format(len(features.columns)))
```

     Number of total features: 17 


## Split Train and Test dataset

Now that we have normalize the numerical values, we can move to the next step which is splitting the data into train and test dataset.


```python
X_train, X_test, y_train, y_test = train_test_split(features, target,test_size = 0.4, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```




    ((89283, 17), (59522, 17), (89283,), (59522,))


from sklearn.model_selection import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    target, 
                                                    test_size = 0.20, 
                                                    random_state = 42,
                                                   )

# Display result after splitting..
print("results of the split\n------")
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

print("\nclass distribution\n------")
print('y_train class distribution')
print(y_train.value_counts(normalize=True))
print('y_test class distribution')
print(y_test.value_counts(normalize=True))




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 148805 entries, 0 to 148804
    Data columns (total 18 columns):
     #   Column           Non-Null Count   Dtype  
    ---  ------           --------------   -----  
     0   age              148805 non-null  float64
     1   income           148805 non-null  float64
     2   time             148805 non-null  float64
     3   offer_completed  148805 non-null  int64  
     4   transaction      148805 non-null  int64  
     5   reward           148805 non-null  float64
     6   difficulty       148805 non-null  float64
     7   duration         148805 non-null  float64
     8   bogo             148805 non-null  float64
     9   discount         148805 non-null  float64
     10  informational    148805 non-null  float64
     11  email            148805 non-null  float64
     12  mobile           148805 non-null  float64
     13  social           148805 non-null  float64
     14  web              148805 non-null  float64
     15  gender_F         148805 non-null  int64  
     16  gender_M         148805 non-null  int64  
     17  gender_O         148805 non-null  int64  
    dtypes: float64(13), int64(5)
    memory usage: 21.6 MB

accuracy = accuracy_score(y_train,np.ones(len(y_train)))
fscore = fbeta_score(y_train,np.ones(len(y_train)), beta=0.5)

# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, fscore: {:.4f}]".format(accuracy, fscore))

Since the offers_completed in the dataset is really imbalanced and is very underrepresented, we will be using the F1-score as our metric. F1-score is the combination of precision and recall in single unit.

## Modelling Data

### Decision Tree Classifier


```python
start = time()
dtc_model = DecisionTreeClassifier()
dtc_model.fit(X_train, y_train)
dtc_pred = dtc_model.predict(X_test)
end = time()
dtc_total_time = end - start
accuracy_dtc = accuracy_score(y_test, dtc_pred)

print("Classification Report: \n", classification_report(y_test, dtc_pred))
```

    Classification Report: 
                   precision    recall  f1-score   support
    
               0       0.85      0.86      0.86     46563
               1       0.48      0.47      0.48     12959
    
        accuracy                           0.77     59522
       macro avg       0.67      0.67      0.67     59522
    weighted avg       0.77      0.77      0.77     59522
    


### Random Forest Classifier


```python
start = time()
rfc_model = RandomForestClassifier()
rfc_model.fit(X_train, y_train)
rfc_pred = rfc_model.predict(X_test)
end = time()
rfc_total_time = end - start
accuracy_rfc = accuracy_score(y_test, rfc_pred)

print("Classification Report: \n", classification_report(y_test, rfc_pred))
```

    Classification Report: 
                   precision    recall  f1-score   support
    
               0       0.82      0.90      0.85     46563
               1       0.43      0.28      0.34     12959
    
        accuracy                           0.76     59522
       macro avg       0.62      0.59      0.60     59522
    weighted avg       0.73      0.76      0.74     59522
    


### Ada Boost Classifier


```python
start = time() 
abc_model = AdaBoostClassifier()
abc_model.fit(X_train, y_train)
abc_pred = abc_model.predict(X_test)
end = time() 
abc_total_time = end-start
accuracy_abc = accuracy_score(y_test, abc_pred)

print('Classification Report: \n', classification_report(y_test, abc_pred))
```

    /opt/homebrew/Caskroom/miniconda/base/envs/starbuckproject/lib/python3.12/site-packages/sklearn/ensemble/_weight_boosting.py:519: FutureWarning: The SAMME.R algorithm (the default) is deprecated and will be removed in 1.6. Use the SAMME algorithm to circumvent this warning.
      warnings.warn(


    Classification Report: 
                   precision    recall  f1-score   support
    
               0       0.81      0.96      0.88     46563
               1       0.62      0.21      0.31     12959
    
        accuracy                           0.80     59522
       macro avg       0.72      0.59      0.60     59522
    weighted avg       0.77      0.80      0.76     59522
    


### K-Neighbors Classifier


```python
start = time() 
knc_model = KNeighborsClassifier(n_neighbors=5)
knc_model.fit(X_train, y_train)
knc_pred = knc_model.predict(X_test)
end = time() 
knc_total_time = end-start
accuracy_knc = accuracy_score(y_test, knc_pred)

print('Classification Report: \n', classification_report(y_test, knc_pred))
```

    Classification Report: 
                   precision    recall  f1-score   support
    
               0       0.81      0.94      0.87     46563
               1       0.49      0.22      0.30     12959
    
        accuracy                           0.78     59522
       macro avg       0.65      0.58      0.59     59522
    weighted avg       0.74      0.78      0.75     59522
    


### Gradient Boosting Classifier


```python

start = time() 
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_pred = gbc.predict(X_test)
end = time() 
gbc_total_time = end-start
accuracy_gbc = accuracy_score(y_test, gbc_pred)

print('Classification Report: \n', classification_report(y_test, gbc_pred))
```

    Classification Report: 
                   precision    recall  f1-score   support
    
               0       0.87      0.92      0.89     46563
               1       0.64      0.51      0.57     12959
    
        accuracy                           0.83     59522
       macro avg       0.76      0.71      0.73     59522
    weighted avg       0.82      0.83      0.82     59522
    


## Model Decision

Let's compare all F1-Score and fined the best result among all.


```python
# Defining the data

data = {
    'Model' : ["Decision Tree", "Random Forest","Ada Boost", "K-Neighbors", "Gradient Boosting"],
    'F1-Score' : [f1_score(y_test, dtc_pred), f1_score(y_test, rfc_pred), f1_score(y_test, abc_pred), f1_score(y_test, knc_pred), f1_score(y_test, gbc_pred)],
    'Accuracy' : [accuracy_dtc * 100.0, accuracy_rfc * 100.0, accuracy_abc * 100.0, accuracy_knc * 100.0, accuracy_gbc * 100.0],
    'Training Duration' : [dtc_total_time, rfc_total_time, abc_total_time, knc_total_time, gbc_total_time]
}

# Creating a DataFrame
df_ModelScore = pd.DataFrame(data)

# Formatting the Accuracy column to show percentages
df_ModelScore['Accuracy'] = df_ModelScore['Accuracy'].apply(lambda x: "%.2f%%" % x)

# Sorting by F1-Score in descending order
df_ms_sorted = df_ModelScore.sort_values(by='F1-Score', ascending=False, ignore_index = True)

print(df_ms_sorted)

```

                   Model  F1-Score Accuracy  Training Duration
    0  Gradient Boosting  0.565708   83.06%           4.887916
    1      Decision Tree  0.477332   77.36%           0.269880
    2      Random Forest  0.339083   76.18%           5.695792
    3          Ada Boost  0.310819   79.94%           1.275176
    4        K-Neighbors  0.300188   78.09%           2.689123


The model that scored the highest is Gradient Boosting Classifier with a score of 0.566 and the highest accuracy as well with 83%. While Gradient Boosting can be slower to train compared to some algorithms, it can handle large datasets efficiently and effectively.


```python
#plotting the n-number of features in a bar chart

def feature_plot(importances, X_train, y_train, n=5):
    # Display the five most important features by default
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:n]]
    values = importances[indices][:n]

    # Creat the plot
    fig = plt.figure(figsize = (8,5))
    plt.title(f"Normalized Weights for First {n} Predictive Features")#, fontsize = 16)
    plt.bar(np.arange(n), values, width = 0.2, align="edge", color = ('#E26741',0.7), \
          label = "Feature Weight")
    plt.bar(np.arange(n) - 0.2, np.cumsum(values), width = 0.2, align = "edge", color = ('#8E1FEF',0.7), \
          label = "Cumulative Feature Weight")
    plt.xticks(np.arange(n), columns)
    #pl.xlim((-0.5, 4.5))
    plt.ylabel("Weight")
    plt.xlabel("Feature")#
    
    plt.legend(loc = 'upper left')
    plt.show()
```


```python
#  Extract the feature importances using .feature_importances_ 
importances = gbc.feature_importances_

# Plot
feature_plot(importances, X_train, y_train)
```


    
![png](output_152_0.png)
    


The top five features that that are predicted by our supervised learning model whether a customer would complete the offer or not are plotted in the graph above with the highest weighted feature as time followed by informational,reward,duration and income.

# Conclusion

__Challenges__

There were a few challenges that were faced in this project. Initially we discovered that the data were inconsistent for example for the age. Some of the customer's have very high value in age(118 years old) which could be some default value passed in the system when customer didn't entered their age when registering. This customers also have missing values for other main attributes. So, instead of taking this data in which would cause a bias result, we removed them. The percentage of them were about 13% from the total unique customers that we have. Removing this has reduced our dataset size which reduced our training sets too. 

Other inconsistency in data we found is in th eportfolio dataframe where there are *offer_id* and *offer id* in the same *value* column. So these are the small things that we need to clean up prior to analysing the dataset which if not, it may impact our analysis.

Besides, whenever a customer made a transaction, the transaction could not be link to whether the customer using the voucher or not. Let say if you can get this information, we can trace to the offer and find out which offer bring the higher revenue to the company. 

__Ideas for Improvements__

* There are many other attributes to explore in the EDA part. For example solving queries related to the channels(either web, mobile or social media) used in distributing offers, duration and membership.

* We could also try different modelling algorithms and applying some hyperparameter tuning to get better prediction model and on the same time making sure we are not overfitting our model. 

