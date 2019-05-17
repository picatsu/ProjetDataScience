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

"""Knn"""
print("############Knn############")

"""NaiveBayes"""
print("############NaiveBayes############")

"""Backpropagation = MLP pour Multilayer Perceptron"""
print("############Backpropagation############")

"""LogisticRegression"""
print("############LogisticRegression############")

"""RandomForest"""
print("############RandomForest############")

"""Kernel SVM = Support Vector Machine"""
print("############Kernel SVM############")

"""Tsne"""
print("############Tsne############")
