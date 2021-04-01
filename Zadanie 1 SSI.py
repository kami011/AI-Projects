#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import copy
sns.set_palette('husl')


# In[2]:


iris = pd.read_csv("iris.csv")


# In[3]:


iris.head()


# In[4]:


iris.describe()


# In[5]:


iris.info()
print(iris.iloc[149])


# In[6]:


g = sns.pairplot(iris, hue='variety', markers='+')


# In[7]:


sns.violinplot(y='variety',x="sepal.length",data=iris,inneer='quartile')


# In[8]:


class DataProcessing:
    @staticmethod
    def shuffle(X):
        for i in range(len(X)-1, 0, -1):
            tmp = random.randint(0, i)
            X.iloc[i], X.iloc[tmp] = X.iloc[tmp], X.iloc[i]
        return X
    
    @staticmethod
    def split(X, Y):
        splt = int(0.7 * len(X))
        Xtrain = X.iloc[:splt,:]
        Xval = X.iloc[splt:,:]
        Ytrain = Y.iloc[:splt]
        Yval = Y.iloc[splt:]
        return Xtrain, Xval, Ytrain, Yval
    
    @staticmethod
    def normalize(X):
        result = X.copy()
        for col in X.columns:
            maximum = X[col].max()
            minimum = X[col].min()
            result[col] = (X[col] - minimum) / (maximum - minimum)
        return result


# In[9]:


irisShuffled = DataProcessing.shuffle(iris)
X = irisShuffled.iloc[:,:-1] #wartości
Y = irisShuffled.iloc[:,-1] #klasy
Xnormalized = DataProcessing.normalize(X)
Xtrain, Xval, Ytrain, Yval = DataProcessing.split(Xnormalized, Y)
