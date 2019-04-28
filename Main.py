"""
import csv
csv.register_dialect('myDialect',
delimiter = ',',
quoting=csv.QUOTE_ALL,
skipinitialspace=True)

with open('spambase.data', 'r') as f:
reader = csv.reader(f, dialect='myDialect')
for row in reader:
    print(row[57])
"""
import matplotlib.pyplot as plt
from numpy.core.tests.test_mem_overlap import xrange
from sklearn import preprocessing, metrics
import numpy as np
import pprint

from sklearn.metrics import confusion_matrix
"""Knn"""
print("############Knn############")


"""NaiveBayes"""
print("############NaiveBayes############")


"""Backpropagation"""
print("############Backpropagation############")


"""LogisticRegression"""
print("############LogisticRegression############")


"""RandomForest"""
print("############RandomForest############")


"""Kernel SVM"""
print("############Kernel SVM############")


"""Tsne"""
print("############Tsne############")

# ajouter SVM - Support Vector Machine
