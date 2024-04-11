---
title: "Exploratory Data Analysis on Kaggle Dataset"
date: 2023-08-11T00:11:04+05:30
tags: ["Kaggle Playground"]
draft: false
---
![Main](/docs/eda_kaggle/Main.png)

I have been thinking of tinkering with Kaggle because it looks fun. 

For beginners to get started, Kaggle offers the “Playground Series”. 

These are synthetically generated datasets that are much easier to get started with. 

As of right now, a new dataset called the [Predict CO2 Emissions in Rwanda](https://www.kaggle.com/competitions/playground-series-s3e20)is available. 

I wanted to provide general steps for EDA along with specific advice for this dataset. 

Let’s get started.

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) is a technique through which we can investigate a Dataset. 

There are 6 steps for EDA:
1. Reading the Data
2. Variable Identification
3. Univariate Analysis
4. Bivariate and Multivariate Analysis
5. Missing Values Identification
6. Outlier Identification

As we are all statisticians, we are supposed to understand the problem and create a hypothesis before looking at the data. 

Problem Definition: Accurately monitoring Carbon emissions is critical to fight Climate Change, infrastructure needed for such an endeavor is still being developed in African countries. Is there a precise way to predict future carbon emissions?

I have two hypotheses:

1. Carbon emissions can be accurately predicted by modeling based on Latitude, Longitude, SulphurDioxide, CarbonMonoxide, NitrogenDioxide, Ozone, Formaldehyde. 

2. Carbon emissions exhibit trend and seasonality, these can be then used to model future predictions.

These will be validated with the data and EDA.

### 1. Importing Libraries and Reading the Data

We are importing a few core libraries needed for performing visualizations and data manipulation.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from tqdm.notebook import tqdm
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


pd.options.display.float_format = '{:.5f}'.format
pd.options.display.max_rows = None

%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```

We are seeding this code, so we can reproduce the same results. 

We are seeding with the number `42` because it is the answer to the ultimate question in life, the universe, and everything.

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
```

Now, we will read the data, using pandas function `read_csv`.

```python
DATA_DIR = "/kaggle/input/playground-series-s3e20"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_sub = os.path.join(DATA_DIR, "sample_submission.csv")

train_csv = pd.read_csv(train_path)
test_csv = pd.read_csv(test_path)
sample_sub_csv = pd.read_csv(sample_sub)
```

We have three DataFrames: Train, Test, and Sample Submission

### 2. Variable Identification

The competition rules state that, we need to predict `emission`, which is our target variable. 

We can take a general look at all the variables and see their datatypes using the `.info()` method. 

![1](/docs/eda_kaggle/1.png)

Almost all the features are continuous variables (float and int). 

Which means we need not spend a lot of time processing them. 

### 3. Univariate Analysis
Univariate Analysis is analyzing one feature at a time.

There are two sub-types: Non Graphical methods and Graphical methods.

#### 3.1 Statistical Summary - Non-Graphical Method
A statistical summary gives important information about the data in a sample such as `min`, `max`, `mean`, `std` etc.

In Pandas, use the `describe` function to get a Statistical Summary across the data frame. 

The `include = "all"`, explicitly provides the Statistical Summary for all the variables.

```python
train_csv.describe(include = "all")
```

![2](/docs/eda_kaggle/2.png)

There are a few observations, we can make here:

1. There are three years in the training data: 2019, 2020, and 2021. We can infer this from the `min`, `max`, and `50%` of the `year` column

2. Our target variable, `emission`, is in between the range `0.0000` and `3167.76800`. It should be skewed right (the tail is on the right side i.e. more values on the left). These can be inferred from the `emission` column.

#### 3.2 Graphical Method

We are going to visualize two variables: `emission` and `year` individually. 

##### 3.2.1 Univariate Analysis of `emission` column
We will be plotting the histogram to visualize the distribution of the `emission` column. 

![3](/docs/eda_kaggle/3.png)

We can see that the data is concentrated on the left side of the `emission` axis.

Let us add a line to make it a bit more clear.

```python
sns.set_style("darkgrid")
plt.figure(figsize = (13, 7))
sns.histplot(train_csv["emission"], kde = True, bins= 15)
plt.title("Emission Distribution", fontsize = 15)
display(plt.show(), train_csv["emission"].skew())
```

![4](/docs/eda_kaggle/4.png)

When Kernel Density Estimate (KDE) is set to `True`, it will smooth the distribution. 

Pandas has the `skew()` function which calculates skew for every column.

Calculating `skew()` for the `emission` column, we can observe that the skew is Positive. 

It indicates that the distribution  is asymmetrical  and the tail is larger towards the right-hand side of the distribution

##### 3.2.2 Univariate Analysis of `year` column
We will be observing the count of the `year` column across the years. 

```python
plt.figure(figsize = (13, 7))
sns.countplot(x = "year", data = train_csv)
plt.title("Distribution across years")
plt.show()
```

![5](/docs/eda_kaggle/5.png)

To look at the actual number:

```python
train_csv["year"].value_counts()
```

![6](/docs/eda_kaggle/6.png)

We can observe that across the years distribution is uniform. 

Looking at the `year` column in the test dataset

```python
plt.figure(figsize = (4, 7))
sns.countplot(x = "year", data = test_csv)
plt.title("Distribution in 2022")
plt.show()
```

![7](/docs/eda_kaggle/7.png)

```python
test_csv["year"].value_counts()
```

Looking at the count of `year` in the test dataset, we can observe that the value is lesser than other years.

### 4. Bivariate and Multivariate Analysis
Multivariate Analysis is observing two(Bivariate) or more features and determining the empirical relationship between them. 

#### 4.1 Between `emission` and `year`
We will first observe the mean `emission` for all the years.

This would help us in comparing the `emission` for three years and in coming to a conclusion on which year has the highest `emission`.

```python
train_csv.groupby("year")["emission"].mean()
```

![8](/docs/eda_kaggle/8.png)

We can view this `emission` as a function of time split between the years. 

```python
sns.set_style("darkgrid")
plt.figure(figsize = (13, 7))
plot = sns.relplot(data = train_csv, x = "week_no", y = "emission", col = "year", kind = "line", legend = False)
plot.figure.subplots_adjust(top=0.8);
plot.fig.suptitle("Bivariate Analysis between emission and year")
plt.show()
```

![9](/docs/eda_kaggle/9.png)

From this observation, we can observe that:

1. Emission seems to be spiking between weeks `10` to `20` and `40` to `45`, irrespective of the year
   
2. Emission of the year `2019` is very similar to `2021` 
   
3. Emission of the year `2020` is lesser than the other years, which could be a factor of COVID.


I am going to plot the correlation maps for SulphurDioxide features and Carbon Monoxide features.

```python
sulphur_list = [x for x in train_csv.columns if "SulphurDioxide" in x]
correlation_so2 = train_csv[sulphur_list + ["emission"]].corr()
sns.heatmap(correlation_so2, annot = True)
plt.title("Correlation Analysis between Sulphur Dioxide and Emission", fontsize = 15)
plt.show()
```

![10](/docs/eda_kaggle/9.png)

We can observe a correlation between SulphurDioxide features and Emission

```python
carbon_list = [x for x in train_csv.columns if "CarbonMonoxide" in x]
correlation_co = train_csv[carbon_list + ["emission"]].corr()
sns.heatmap(correlation_co, annot = True)
plt.title("Correlation Analysis between Carbon Monoxide and Emission", fontsize = 15)
plt.show()
```

![11](/docs/eda_kaggle/11.png)

We can observe a similar correlation between SulphurDioxide features and Emission. 

This trend also continues to other features such as NitrogenDioxide, Formaldehyde, etc. 

### 5. Missing Values Identification
We are going to check for Null values among Features. 

```python
null_result = train_csv.isna().sum().sort_values(ascending = True).plot(kind = "barh", figsize = (23, 23))
```

![12](/docs/eda_kaggle/12.png)

There are more null values in some features than data. 

We will deal with such features while modeling. 

### 6. Outlier Values Identification
We will search for outliers in our target column. 

```python
# Outliers through BoxPlot
sns.set_style("darkgrid")
plt.figure(figsize = (13,7))
sns.boxplot(train_csv["emission"])
plt.title("Outliers in Emission", fontsize = 15)
plt.show()
```

![13](/docs/eda_kaggle/13.png)

We have reached the following observation from this EDA:

1. There are three years in the training data: 2019, 2020, and 2021. 
2. Our target variable, `emission`, is in between the range `0.0000` and `3167.76800`. It is skewed right.
3. Across all the years in the training dataset, the emission is distributed equally. 
4. In the Test dataset, the emission values are lesser than the training dataset
5. Emission seems to be spiking between weeks `10` to `20` and `40` to `45`, irrespective of the year
6. Emission of the year `2019` is very similar to `2021` 
7. Emission of the year `2020` is lesser than the other years, which could be a factor of COVID.
8. There is a correlation between SulphurDioxide and emission
9. There is a correlation between CarbonMonoxide and emission
10. There are more null values in some features than data points themselves.
11. There are a significant amount of outliers in our target column. 


These validate our hypothesis. 

We will be using these observations to create a model. 

Stay tuned!
