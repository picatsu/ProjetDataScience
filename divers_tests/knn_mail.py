# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:01:03 2019

@author: Zylv1

Test du KNN pour les mails, donc avec les vraies données de Wandrille

Référence super utile :
    https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

Il y a 57 champs, et 1 champ pour décrire si le mail est un spam ou non

"""

import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd 


# import some data to play with

# Depuis la librairie de Python
'''iris = datasets.load_iris()
X = iris.data
y = iris.target'''

csvValuesColumnNumber = 57

# Depuis un csv
csvFilePath = "../spambase/spambase.data";
mailDataset = pd.read_csv(csvFilePath, header = None)  # names=names,

dataFieldsValues = mailDataset.iloc[:, :-1].values  # : signifie "tout" -> :-1 signifie "toutes les colonnes sauf la dernière"
dataLabels = mailDataset.iloc[:, csvValuesColumnNumber].values  

X_train, X_test, y_train, y_test = train_test_split(dataFieldsValues, dataLabels, test_size=0.2, shuffle=True) # test_size = 1 - train_size

#print(dataFieldsValues)
#print(X_train)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=4)  
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test) 

#print(y_pred)  # 0 correspond to Versicolor, 1 to Verginica and 2 to Setosa

#jprint(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


minKNumber = 1
maxKNumber = 10

error = []

# Calculating error for K values between minKNumber and maxKNumber
for i in list(range(minKNumber, maxKNumber)):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    meanRatio = np.mean(pred_i != y_test)
    print("meanRatio = " + str(meanRatio) + " for k = " + str(i))
    error.append(meanRatio)
    
plt.figure(figsize=(14, 6))  
plt.plot(list(range(minKNumber, maxKNumber)), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Taux d\'erreur en fonction de la valeur de K')  
plt.xlabel('Valeur de K')  
plt.ylabel('Erreur moyenne') 