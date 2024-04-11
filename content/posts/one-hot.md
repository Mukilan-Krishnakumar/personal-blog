---
title: "Do you really 'One Hot' Encode?"
date: 2023-08-11T06:50:14+05:30
draft: false
---
![Main](/docs/one-hot/Main.png)

I was working on a classifier from scratch for the [Ham Spam Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). It is a dataset which has messages column along with subsequent value classifying it as Ham or Spam. 

I would like to think that it is called “Ham” referring to the juiciness of an email. Juicier the email, better the quality. No puns intended. 

We all know what a spam is, I even got one 3 weeks back from a Cyprus Military Man. 

![1](/docs/one-hot/1.png)

Neural networks cannot process text as it is. Our categories are essentially strings. We need to convert them to numbers. 

Converting Categorical Variables (Fancy term for our categories : Ham and Spam) to numerical values is called Encoding. 

There are two types of Encoding:
1. Ordinal Encoding
2. One Hot Encoding

Ordinal Encoding is really simple. If the categories can be ranked, then we can give them numbers and arrange them. 

Consider the categories: “Low”, “Medium”, “High”. 

We can then give them numerical values correlated to their categories like:

![2](/docs/one-hot/2.png)

But, we have Ham and Spam. They don’t have any natural ranking. So we are going to go with the other Encoding called “One Hot Encoding”.

One Hot Encoding is a very simple way of representing categories. It only has two values: "True" (represented by 1) and “False” (represented by 0).

At any moment in a record, only one value is True. It is much more easier to understand visually. 

Let us take our two categories: Ham and Spam. Let us create two columns and call them “Ham” and “Spam”. 

![3](/docs/one-hot/3.png)

That’s it.

You can use `sklearn.preprocessing.OneHotEncoder`, but that would mean not getting our hands dirty. Only when we build our own things we can appreciate beauty. 

Never use an abstraction just because, build it from scratch, break it and then move on to abstraction. Or else you will never truly understand it. 

First, we need to find the categories in the given column. We will be dealing with Dataframes and pandas provide this neat function called `unique` which gives us the number of unique values. You can also use `set` function to find the number of unique features but that would increase time complexity. 

```python
uniq_values = df_train[df_column].unique()
```

We need the number of unique values:

```python
number_of_features = len(uniq_values)
```

We will create a dictionary which will store our individual categorical representations, which will later be transformed into Pandas DataFrame. We will also initialize a list of zeros, which will then be sent to the representations. 

```python
main_dict = {}
zero_list = np.zeros(len(df))
for val in uniq_values:
	main_dict[val] = list(zero_list)
```

Notice that my code says `list(zero_list)` and not just `zero_list`. The first case would create a new list and the latter would just reference to the same one (if you change something with one variable the other changes too). 

How do I know that? Maybe because I spent half hour head butting into this dumb issue. 

Now, go through the origin column values and assign “1” to corresponding categorical representations in our dictionary. 

```python
for num, record in enumerate(df[df_column]):
	main_dict[str(record)][num] = 1.0
```

Convert this dictionary to a DataFrame:

```python
df_one_hot = pd.DataFrame.from_dict(main_dict)
```

We can package this better by porting all the code into a function and returning the updated DataFrame.

```python
def OneHotEncoder(df, df_column):
    uniq_values = df[df_column].unique()
    number_of_features = len(uniq_values)
    main_dict = {}
    zero_list = np.zeros(len(df))
    for val in uniq_values:
        main_dict[val] = list(zero_list)
    for num, record in enumerate(df[df_column]):
        main_dict[str(record)][num] = 1.0
    df_one_hot = pd.DataFrame.from_dict(main_dict)
    frames = [df, df_one_hot]
    df_result = pd.concat(frames,axis=1, join='inner')
    return df_result
```

That’s it. A very simple idea and a very simple function which goes a long way in your ML journey. 

This is actually part of the classifier I am building from scratch in PyTorch. If you are interested to learn, how to do that, visit this GitHub repository. 

Stay hydrated!
