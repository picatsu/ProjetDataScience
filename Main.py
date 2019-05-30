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
import Algos.KNearestNeighbors as KNN
import Algos.LogisticRegression as LR
import Algos.RandomForest as RF
import Algos.SupportVectorMachine as SVM

"""Knn"""
KNN.test()

"""NaiveBayes"""
print("############NaiveBayes############")

"""Backpropagation = MLP pour Multilayer Perceptron"""
print("############Backpropagation############")

"""LogisticRegression"""
LR.test()

"""RandomForest"""
RF.test()

"""Kernel SVM = Support Vector Machine"""
SVM.test()

"""Tsne"""
print("############Tsne############")
