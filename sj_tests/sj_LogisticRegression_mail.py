# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:01:03 2019

@author: Zylv1

Logistic Regression pour les mails, donc avec les vraies données de Wandrille

Très bons résultats, et très rapide

Référence super utile :
    https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

Il y a 57 champs, et 1 champ pour décrire si le mail est un spam ou non

"""

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd 
from sklearn.linear_model import LogisticRegression  


csvValuesColumnNumber = 57

# Depuis un csv
csvFilePath = "../spambase/spambase.data";
mailDataset = pd.read_csv(csvFilePath, header = None)  # names=names,

dataFieldsValues = mailDataset.iloc[:, :-1].values  # : signifie "tout" -> :-1 signifie "toutes les colonnes sauf la dernière"
dataLabels = mailDataset.iloc[:, csvValuesColumnNumber].values

X = dataFieldsValues
y = dataLabels

X_train, X_test, y_train, y_test = train_test_split(dataFieldsValues, dataLabels, test_size=0.2, shuffle=True) # test_size = 1 - train_size
# Inutile d'utiliser de StandardScaler ici

LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=100000)
LR.fit(X_train, y_train) # LR -> classifier

y_pred = LR.predict(X_test) 

meanRatio = np.mean(y_pred != y_test)
print(meanRatio)

# Identique, juste fait avec la fonction spécifique à LogisticRegression :
#meanRatoVerif = 1 - round(LR.score(X_test, y_test), 4)
#print(meanRatoVerif)

print(classification_report(y_test, y_pred))

predictionRatio = meanRatio

errorArray = [predictionRatio, predictionRatio]
rangeArray = [1, 2]

plt.figure(figsize=(14, 6))  
plt.plot(rangeArray, errorArray, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Taux d\'erreur en fonction de la valeur de K')  
plt.xlabel('Valeur de K')  
plt.ylabel('Erreur moyenne') 
