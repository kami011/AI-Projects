#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import copy


# In[2]:


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
        for col in X[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]:
            maximum = X[col].max()
            minimum = X[col].min()
            result[col] = (X[col] - minimum) / (maximum - minimum)
        return result

class SoftSet:
    @staticmethod
    def clasification(X, Y, sample):
        results = []
        for i in range(0, len(X), 1):
            tmp = 0
            for k in Y[i]:
                tmp += Y[i][k] * sample[k]
            results.append(tmp)
        index = results.index(max(results))
        return X[index]
    
    @staticmethod
    def irisCharacterization(X, av): #Funkcja przydziela odpowiednie atrybuty, na podstawie średnich z danych kolumn i porównywania
        attributes = {'sepal.length<av':0, 'sepal.length>av':0, 'sepal.width<av':0, 'sepal.width>av':0, 'petal.length<av':0, 'petal.length>av':0, 'petal.width<av':0, 'petal.width>av':0}
        names = ['sepal.length<av', 'sepal.length>av', 'sepal.width<av', 'sepal.width>av', 'petal.length<av', 'petal.length>av', 'petal.width<av', 'petal.width>av']
        majority = int(0.3 * len(X))
        for row in X.iloc():
            i = 0
            for j in row:
                if isinstance(j, str)!=True:
                    if j < float(av[i]):
                        name = names[i*2]
                        attributes[name]=attributes[name]+1
                    else:
                        name = names[i*2+1]
                        attributes[name]=attributes[name]+1
                    i+=1
        for name in names:
            if attributes[name]>=majority:
                attributes[name]=1
            else:
                attributes[name]=0
        
        return attributes


X = ["marchewka", "papryka ostra", "papryka słodka", "pomidor", "mango", "burak", "sałata lodowa"]
Y = [{'świeże':1, 'mrożone':0, 'ostre':0, 'słodkie':1, 'zielone':0, 'czerwone':0, 'lokalne':1, 'tropikalne':0, 'liściaste':0, 'bulwowe':0},
    {'świeże':1, 'mrożone':0, 'ostre':1, 'słodkie':0, 'zielone':0, 'czerwone':1, 'lokalne':1, 'tropikalne':0, 'liściaste':0, 'bulwowe':0},
     {'świeże':1, 'mrożone':0, 'ostre':0, 'słodkie':1, 'zielone':0, 'czerwone':1, 'lokalne':1, 'tropikalne':0, 'liściaste':0, 'bulwowe':0},
    {'świeże':1, 'mrożone':0, 'ostre':0, 'słodkie':1, 'zielone':0, 'czerwone':1, 'lokalne':0, 'tropikalne':0, 'liściaste':0, 'bulwowe':0},
    {'świeże':0, 'mrożone':1, 'ostre':0, 'słodkie':1, 'zielone':0, 'czerwone':0, 'lokalne':0, 'tropikalne':1, 'liściaste':0, 'bulwowe':0},
    {'świeże':1, 'mrożone':0, 'ostre':0, 'słodkie':0, 'zielone':0, 'czerwone':0, 'lokalne':1, 'tropikalne':0, 'liściaste':0, 'bulwowe':1},
    {'świeże':1, 'mrożone':0, 'ostre':0, 'słodkie':0, 'zielone':1, 'czerwone':0, 'lokalne':1, 'tropikalne':0, 'liściaste':1, 'bulwowe':0}]
print("A:", SoftSet.clasification(X, Y, {'świeże':0.8, 'mrożone':0, 'ostre':0.6, 'słodkie':0, 'zielone':0, 'czerwone':0.4, 'lokalne':0, 'tropikalne':0, 'liściaste':0, 'bulwowe':0}))
print("B:", SoftSet.clasification(X, Y, {'świeże':0, 'mrożone':0.6, 'ostre':0, 'słodkie':0.6, 'zielone':0.7, 'czerwone':0, 'lokalne':0, 'tropikalne':0, 'liściaste':0.5, 'bulwowe':0}))
print("C:", SoftSet.clasification(X, Y, {'świeże':1, 'mrożone':0, 'ostre':0, 'słodkie':1, 'zielone':1, 'czerwone':1, 'lokalne':0, 'tropikalne':0, 'liściaste':0, 'bulwowe':0}))


# In[3]:


iris = pd.read_csv("iris.csv")
irisNormalized = DataProcessing.normalize(iris)
average = irisNormalized.mean()
setosadf = irisNormalized.loc[irisNormalized.variety=='Setosa']
versicolordf = irisNormalized.loc[irisNormalized.variety=='Versicolor']
virginicadf = irisNormalized.loc[irisNormalized.variety=='Virginica']
Y1 = SoftSet.irisCharacterization(setosadf, average)
Y2 = SoftSet.irisCharacterization(virginicadf, average)
Y3 = SoftSet.irisCharacterization(versicolordf, average)
X = ['setosa', 'virginica', 'versicolor']
Y = [Y1, Y2, Y3]
print("Wartości atrybutów dla kolejno setosy, virginiki, versicolor:")
print(Y)
print("\n Przykładowy wynik klasyfikacji irysów:", SoftSet.clasification(X, Y, {'sepal.length<av':0.1, 'sepal.length>av':0.5, 'sepal.width<av':0.2, 'sepal.width>av':0.3, 'petal.length<av':0.5, 'petal.length>av':0.1, 'petal.width<av':0.8, 'petal.width>av':0}))

