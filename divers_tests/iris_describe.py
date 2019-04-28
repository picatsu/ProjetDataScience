# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:34:30
 2018

@author: Wandrille
"""

from sklearn import datasets
import pandas as pd

# Loading IRIS dataset from scikit-learn object into iris variable.
iris = datasets.load_iris()

print("==========================")

# Prints the type/type object of iris
print("Type d'objet géré par Iris : " + type(iris).__name__)
# <class 'sklearn.datasets.base.Bunch'>

# prints the dictionary keys of iris data
print("")
print("iris.keys() : ")
print(iris.keys())

# prints the type/type object of given attributes
print(type(iris.data), type(iris.target))

# prints the no of rows and columns in the dataset
print(iris.data.shape)

# prints the target set of the data
print(iris.target_names)

# Load iris training dataset
X = iris.data

# Load iris target set
Y = iris.target

# Convert datasets' type into dataframe
df = pd.DataFrame(X, columns=iris.feature_names)

# Print the first five tuples of dataframe.
print(df.head())
