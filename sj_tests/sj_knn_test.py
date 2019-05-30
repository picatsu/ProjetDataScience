# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 12:39:41 2019
@author: Zylv1


Implémentation de l'algo KNN, repris du site :
  https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

C'est le même jeu de données que celui utilisé par Wandrille dans le fichier knn_iris_full.py


"""

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 
from sklearn.model_selection import train_test_split  

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

csvFilePath = "datasets/iris.data";


# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisDataset = pd.read_csv(csvFilePath, names=names, header = None)  

X = irisDataset.iloc[:, :-1].values  # : signifie "tout" -> :-1 signifie "toutes les colonnes sauf la dernière"
y = irisDataset.iloc[:, 4].values  

print(irisDataset.head(10))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  

