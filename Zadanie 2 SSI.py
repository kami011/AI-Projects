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


# In[10]:


class KNN:
    @staticmethod
    def MinkowskiMetric(v1, v2, m):
        dim = len(v1)
        distance = 0
        for i in range(dim):
            distance+=abs(v1.iloc[i] - v2.iloc[i])**m
        distance=distance**(1/m)
        return distance
    
    @staticmethod
    def slice(tab, tab2, tab3, start, end):
        i = (start - 1)
        p = tab[end]
        for j in range(start, end):
            if tab[j] < p:
                i = i + 1
                tab[i], tab[j] = tab[j], tab[i]
                tab2.iloc[i], tab2.iloc[j] = tab2.iloc[j], tab2.iloc[i]
                tab3.iloc[i], tab3.iloc[j] = tab3.iloc[j], tab3.iloc[i]
        tab[i + 1], tab[end] = tab[end], tab[i + 1]
        tab2.iloc[i + 1], tab2.iloc[end] = tab2.iloc[end], tab2.iloc[i + 1]
        tab3.iloc[i + 1], tab3.iloc[end] = tab3.iloc[end], tab3.iloc[i + 1]
        return (i + 1)

    @staticmethod
    def quicksort(tab, tab2, tab3, start, end):
        if start < end:
            p = KNN.slice(tab, tab2, tab3, start, end)
            KNN.quicksort(tab, tab2, tab3, start, p - 1)
            KNN.quicksort(tab, tab2, tab3, p + 1, end)
    
    @staticmethod
    def clustering(testSample, Xtrain, Ytrain, k):
        #oblicznie odległości
        classes={'Setosa': 0, 'Versicolor': 0, 'Virginica': 0}
        distances = []
        for i in Xtrain.iloc:
            distances.append(KNN.MinkowskiMetric(testSample,i,2))
        #sortowanie po tablicy distances + zamiana elementów w X i Ytrain
        KNN.quicksort(distances, Ytrain, Xtrain, 0, len(distances)-1)
        #glosowanie
        for i in range(0, k, 1):
            classes[Ytrain[i]] = classes[Ytrain[i]]+1
        return max(classes, key=classes.get)    
    
    @staticmethod
    def check(Xtrain, Ytrain, Xval, Yval, k):
        correct=0
        tmp=0
        for testSample, variety in zip(Xval.iloc, Yval.iloc):
            Xtrain1=copy.copy(Xtrain)
            Ytrain1=copy.copy(Ytrain)
            tmp = KNN.clustering(testSample, Xtrain1, Ytrain1, k) 
            if tmp == variety:
                correct+=1
        accuracy=correct/len(Yval)*100
        print("Accuracy for k = ", k ,": ", accuracy, "CORRECT: ", correct)


# In[11]:


Xtrain1=copy.copy(Xtrain)
Ytrain1=copy.copy(Ytrain)
Xtrain2=copy.copy(Xtrain)
Ytrain2=copy.copy(Ytrain)
Xtrain3=copy.copy(Xtrain)
Ytrain3=copy.copy(Ytrain)
KNN.check(Xtrain1, Ytrain1, Xval, Yval, 2)
KNN.check(Xtrain2, Ytrain2, Xval, Yval, 3)
KNN.check(Xtrain3, Ytrain3, Xval, Yval, 4)

