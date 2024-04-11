---
title: "Building Recurrent Neural Network from scratch in PyTorch"
date: 2023-08-31T06:11:30+05:30
draft: false
layout: single
tags: ['Paper Implementations in PyTorch']
---

![0](/docs/pytorch-rnn/0.png)

Recurrent Neural Networks (RNN) were first introduced in the paper “Finding Structure in Time” by the Psycholinguist, Jeffrey Elmann. 

Building RNN in PyTorch has been made tremendously easy by the abstraction layer providing the RNN Class. 

But if you are anything like me, you would like to deeply understand the network and would rather build it yourself. 

Self-induced suffering is often a precursor to Masochism. Just saying.

That is exactly why we are pursuing this series, not for masochism but for self-reliance. It is aptly named: Paper Implementations in PyTorch.

> TL: DR; 
> This [GitHub repository](https://github.com/Mukilan-Krishnakumar/PyTorch) has the code for Elmann RNN. 
> Consider starring the repository if you found it of any value. 

Let us get started!

## Key Takeaways from the Research Paper

### 1. Problem with Time

Representing time has always been a bit tricky problem. 

Before the introduction of Elmann RNN, time was represented explicitly, along a dimension, where the first temporal event is the first element, and so on. 

A simple example would be taking your dog out for a walk. 

The first temporal event is the dog taking a dump on a stranger’s lawn. 

The second temporal event is you realizing what he is doing. 

The third temporal event is you both running away from the house and so on.

This representation interlocks the size of the pattern i.e. demands every input vector to be the same length. 

### 2. Networks with Memory

Elmann represented time by the effect it had on processing/memory.

He proposed the, then novel, Recurrent Neural Network: A simple Neural Network with a single input layer, a hidden layer, and an output layer. 

Time is represented implicitly and shared through hidden states. 

Hidden states, true to their name are never exposed. 

Consider these states as undercover agents.

## Illustrative Recurrent Neural Network
![1](/docs/pytorch-rnn/1.png)
This is the illustration of a single RNN, rolled onto itself. 

In a typical RNN, a lot of RNN cells are repeated based on the input size. 

We can visualize it by sending in an input text, such as “Je suis ingénieur”. 

Does the input text convey the truth?

What is truth anyway? Why are we put on this earth? Do I really exist?

Don’t stray off, getting back to RNNs. 
![2](/docs/pytorch-rnn/2.png)

The input text is split into words and then vectorized. 

These vectors are then fed into RNN cells at different time steps. 

A wonderful feature of RNN is its ability to handle any length. 

## Building RNNs from Scratch in PyTorch

We are working on the Ham Spam Dataset. 

There is a message column and a target column. 

Messages are classified as Spam or Ham (Not Spam).

It is called Ham because it contains “Juicy” text. I am not kidding.

Before starting any ML project, we need to plan.

We need a plan to conquer, as the great knight Don Quixote once said “Veni, Vidi, Vici”.

This is my game plan:
1. Importing Necessary Libraries
2. Data Preprocessing
3. Vocabulary
4. Dataset and DataLoader
5. Recurrent Neural Networks from Scratch
6. Training Loop
7. Testing Loop

Let’s get started!

### 1. Importing Necessary Libraries

```python
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import re
import wandb
```

We are importing `torch` and associated `torch` libraries to build and train the classifier. 

We are importing `matplotlib` to visualize the dataset if necessary.

We are importing `numpy` and `pandas` for Data Manipulation. 

We are importing `re` to perform regex operations. 

Finally, we are importing `wandb` to track our model experiments. 

```python
wandb.init(project="PyTorch-Paper-Implementations")
```

After initializing the project, let us set the configuration dictionary for this specific run. 

```python
config = {"learning_rate": 1e-3,
          "batch_size": 64,
          "epochs": 40}

wandb.config = config
```

If you have a GPU, set the device to GPU. 

I happen to have MPS, the GPU for Apple machines. 

```python
device = "mps"
```

### 2. Data Preprocessing
Let us perform a quick Exploratory Data Analysis on this dataset. 

Link to the [Kaggle dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

```python
df = pd.read_csv("spam.csv", encoding = "ISO-8859-1")
df.head()
```
![3](/docs/pytorch-rnn/3.png)

We observe unwanted columns in the rear end of the dataset. 

We will drop them.

```python
df.drop(["Unnamed: 2","Unnamed: 3", "Unnamed: 4"], axis = 1, inplace=True)
```

Checking for null values.

```python
df.isna().sum()
```

There are no null values, let us rename the columns to add more clarity.

```python
df.columns = ["target", "message"]
```

Numerically encoding our target column:

```python
conversion_dict = {"ham": 0, "spam": 1}
def conversion_fn(target_val):
    return conversion_dict[target_val]
```

```python
df["target"] = df["target"].map(lambda x: conversion_fn(x))
df.head()
```
![4](/docs/pytorch-rnn/4.png)

Amazing!

Now, we can check for the distribution of data classes.

```python
plt.pie(df["target"].value_counts(), labels= df["target"].unique().tolist(), autopct= '%1.1f%%')
plt.show()
```
![5](/docs/pytorch-rnn/5.png)

The ratio between Ham: Spam is almost 5:1.

We are going to normalize this distribution. 

```python
n = df["target"].value_counts().min()
df =  df.groupby("target").head(n)
plt.pie(df["target"].value_counts(), labels= df["target"].unique().tolist(), autopct= '%1.1f%%')
plt.show()
```
![6](/docs/pytorch-rnn/6.png)

If we import the processing output, we will run through Key Errors. 

To avoid this, I am going to save this and import it again. 

```python
df.to_csv("hamspam_processed.csv", index= False)
train_df = pd.read_csv("hamspam_processed.csv")
```

### 3. Vocabulary 

Vocabulary is the word lookup dictionary. 

It has two components:
1. `token_to_idx` -  Given a token(word), returns the index of token.
2. `idx_to_token` - Given the index, returns the token.

These messages which are being fed to the Vocabulary class, often have unwanted characters which can be removed by Regex. 

Another major functionality in Vocabulary class is Vectorization. 

In this case, we are performing sparse One Hot Vectorization. 

If you would like to learn more, click on this [link](https://mukilank.com/posts/one-hot/). 

```python
class Vocabulary:
    def __init__(self, messages):
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.messages = messages
        self.add_token("<UNK>")
        self.special_char = re.compile(r'[;\\/,!.:*?\"<>|&\']')
        for message in messages:
            for word in message.split(" "):
                word = re.sub(self.special_char, " ", word)
                word = word.lower()
                self.add_token(word)
        
    def add_token(self,token):
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
            
    def vectorize(self, message):
        one_hot = torch.zeros(100,1, len(self.token_to_idx))
        for num, word in enumerate(message.split(" ")):
            word = re.sub(self.special_char, " ", word)
            word = word.lower()
            if num >= 100:
                break
            elif word in self.token_to_idx:
                one_hot[num][0][self.token_to_idx[word]] = 1
            else:
                word = "<UNK>"
                one_hot[num][0][self.token_to_idx[word]] = 1
        return one_hot
    
    def len_token_idx(self):
        return len(self.token_to_idx)
```

### 4. Dataset and DataLoader
Dataset object stores samples and their corresponding labels. 

Keeping this in mind, let us define our Custom Dataset. 

```python
class SpamDataset(Dataset):
    def __init__(self, df, messages_col, target_col, transform = None):
        self.df = df
        self.transform = transform
        
        self.messages = self.df[messages_col]
        self.target = self.df[target_col]
        
        self.vocab = Vocabulary(self.messages)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        message = self.messages[index]
        target = self.target[index]
        
        if self.transform is not None:
            message = self.transform(message)
            
        vectorized_message = torch.tensor(self.vocab.vectorize(message))
        vectorized_target = torch.tensor(target)
        
        return vectorized_message, vectorized_target
    
    def len_token_idx(self):
        return self.vocab.len_token_idx()
```

Wrapping the DataLoader around our Dataset class

```python
train_dataset = SpamDataset(train_df, "message", "target")
train_dataloader = DataLoader(train_dataset, batch_size= config["batch_size"], shuffle=True)
```

### 5. Recurrent Neural Networks from Scratch
Visualizing a Neural Network will help us in designing it. 

![7](/docs/pytorch-rnn/7.png)

Two inputs are fed into the Cell: Actual Text Input and Previous Hidden Input. 

We will combine these and store them as `combined`.

From the `combined` layer, we can deviate into two Linear Layers: i2o (input-to-output) and i2h(input-to-hidden). 

The output from i2o is then fed into the softmax layer for calculating the `output`.

Alright, time to build!

```python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size ,output_size)
        self.softmax = nn.LogSoftmax(dim = 0)
        
    def forward(self, input_message, hidden):
        combined = torch.cat((input_message, hidden), 0)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(self.hidden_size)
```

We initially set the hidden state to a tensor of zeroes. 

```python
input_size = train_dataset.len_token_idx()
hidden_size = train_dataset.len_token_idx()
output_size = 1

model = RNN(input_size, hidden_size, output_size).to(device)
print(model)

wandb.watch(model)
```

Initializing the model, we move the model to GPU. 

We use `wandb.watch` to collect model gradients and topology. 

```python
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
```

We set the loss function to Binary Cross Entropy Loss (BCELoss) and set the optimizer to SGD. 

### 6. Training Loop
Iterating through the DataLoader, we move the features and target tensors to the GPU. 

We process them through our model and log our losses with W&B. 

```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.float()
        y = y.to(device)
        pred = torch.empty((0), dtype = torch.float32).to(device)
        # Compute prediction and loss
        for sentence in X:
            hidden = model.initHidden().to(device)
            for word_vector in sentence[1]:
                output, hidden = model(word_vector, hidden)
            pred = torch.cat((pred, output), 0)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            wandb.log({"loss" : loss})
```

```python
for t in range(config["epochs"]):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
```
![8](/docs/pytorch-rnn/8.png)

### 7. Testing
Time to test our model with Custom Inputs. 

```python
def test_model(input_text):
    vectorized_input = train_dataset.vocab.vectorize(input_text)
    vectorized_input.to("cpu")
    model.to("cpu")
    hidden = model.initHidden().to("cpu")
    for word_vector in vectorized_input[1]:
        word_vector.to("cpu")
        output, hidden = model(word_vector, hidden)
    return output
```

```python
input_text = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
output = test_model(input_text)
print(output)
```

The RNN is predicting it to be Ham but the actual value is Spam. 

I told you in the beginning, this is masochistic.

Reasons for such problems could be RNNs themselves as they experience Vanishing Gradients. 

We will explore these problems and associated solutions in future posts. 

We learned a lot during this implementation. 

Maybe the old fool was right when he said “If In Doubt, Meriadoc, Always Follow Your Nose”. 

Thanks for reading!!

If you got any value out of this post, consider starring the repository. 

Until next time, peace. 
