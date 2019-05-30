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
import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
#sys.path.append('C:\Users\m6d\OneDrive\Bureau\DANT\DATA SCIENCE\ProjetDataScience\ProjetDataScience\Algos')

import KNearestNeighbors as KNN
import LogisticRegression as LR
import RandomForest as RF
import SupportVectorMachine as SVM

"""Knn"""
KNN.test()

"""NaiveBayes"""
#print("############NaiveBayes############")

"""Backpropagation = MLP pour Multilayer Perceptron"""
#print("############Backpropagation############")

"""LogisticRegression"""
LR.test()

"""RandomForest"""
RF.test()

"""Kernel SVM = Support Vector Machine"""
#SVM.test()

"""Tsne"""
print("############Tsne############")
