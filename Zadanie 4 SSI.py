#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random


# In[2]:


class NaiveBayes:
    @staticmethod
    def mean(col):
        return col.mean()
    
    @staticmethod
    def std(col):
        avg = col.mean()
        suma = 0
        for row in col:
            suma = suma + (row-avg)**2
        res = suma/len(col)
        return res
    
    @staticmethod
    def gauss(v, mean, std):
        exponent=np.exp(-(v-mean)**2/(2*std**2))
        return (1/np.sqrt(2*np.pi*std**2)*exponent)
    
    @staticmethod
    def classify(X, sample):
        setosa = X.loc[X.variety=='Setosa']
        versicolor = X.loc[X.variety=='Versicolor']
        virginica = X.loc[X.variety=='Virginica']
        
        varieties = [setosa, versicolor, virginica]
        means = []
        stds = []
        gauss = []
        results = []
        
        for variety in varieties:
            tmp1 = []
            tmp2 = []
            for col in variety[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]:
                tmpav = NaiveBayes.mean(variety[col])
                tmp1.append(tmpav)
                tmpstd = NaiveBayes.std(variety[col])
                tmp2.append(tmpstd)
            means.append(tmp1)
            stds.append(tmp2)
        
        for i in range(len(stds)):
            j=0
            tmp = []
            for col in variety[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]:
                tmp.append(NaiveBayes.gauss(sample[col], means[i][j], stds[i][j]))
                j+=1
            gauss.append(tmp)
            
        for i in range(len(gauss)):
            tmp = 1
            for j in range(len(gauss[i])):
                tmp*=gauss[i][j]
            tmp = tmp*(1/3)
            results.append(tmp)

        return results.index(max(results))


# In[3]:


X = pd.read_csv("iris.csv")
sample = X.iloc[120]
print(sample)
var = NaiveBayes.classify(X, sample)
if var == 0:
    print('SETOSA')
elif var == 1:
    print('VERSICOLOR')
else:
    print('VIRGINICA')



