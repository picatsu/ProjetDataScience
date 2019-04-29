# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:01:03 2019

@author: Zylv1

Test du Backpropagation pour les mails, donc avec les vraies données de Wandrille
-> Plutôt bon, entre 0.050 et 0.080, 5% à 8% d'erreur

Référence super utile :
    https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/

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


csvValuesColumnNumber = 57

# Depuis un csv
csvFilePath = "../spambase/spambase.data";
mailDataset = pd.read_csv(csvFilePath, header = None)  # names=names,

dataFieldsValues = mailDataset.iloc[:, :-1].values  # : signifie "tout" -> :-1 signifie "toutes les colonnes sauf la dernière"
dataLabels = mailDataset.iloc[:, csvValuesColumnNumber].values  

X_train, X_test, y_train, y_test = train_test_split(dataFieldsValues, dataLabels, test_size=0.2, shuffle=True) # test_size = 1 - train_size

# print des données en entrée
#np.set_printoptions(threshold=np.inf)
#print(X_test)

#scaler = StandardScaler()
#scaler.fit(X_train)

#X_train = scaler.transform(X_train)  
#X_test = scaler.transform(X_test)


from sklearn.neural_network import MLPClassifier  
classifier = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  # mlp -> classifier
classifier.fit(X_train, y_train.ravel())  


y_pred = classifier.predict(X_test)  


#print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

meanRatio = np.mean(y_pred != y_test)

predictionRatio = meanRatio

errorArray = [predictionRatio, predictionRatio]
rangeArray = [1, 2]


plt.figure(figsize=(14, 6))  
plt.plot(rangeArray, errorArray, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Taux d\'erreur en fonction de la valeur de K')  
plt.xlabel('Valeur de K')  
plt.ylabel('Erreur moyenne') 