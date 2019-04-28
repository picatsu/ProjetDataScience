# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28

@author: Zylv1

Implémentation regroupant les différents algorithmes, plus propre que de faire pleins de fichiers différents.

Référence super utile pour Naive Bayes :
    https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn/
-> Ce lien est aussi super utile si on veut créer notre propre base de données à partir de mails complets et non formattés.


Il y a 57 champs, et 1 champ pour décrire si le mail est un spam ou non
 

"""

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
# Seulement utiles pour KNN
#from sklearn.preprocessing import StandardScaler
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd 
# Naive Bayes spécifique
from sklearn.naive_bayes import MultinomialNB


def drawBenchmarkForSingleValue(y_pred, y_test) :
    print(classification_report(y_test, y_pred))
    # Dessin d'un tableau pour pouvoir graphiquement le comparer à KNN
    rangeFirst = [1, 2] # = en python 3,  list(range(1, 3)) #
    #print(rangeFirst)
    #print(y_pred)
    predictedRatio = np.mean(y_pred != y_test)
    error = [predictedRatio, predictedRatio]#[3.542, 8.612]
    
    '''error = []
    error.append(predictedRatio)
    error.append(predictedRatio)'''
    
    print(predictedRatio)
    
    plt.figure(figsize=(14, 6))
    plt.plot(rangeFirst, error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Taux d\'erreur pour Naive Bayes')
    plt.xlabel('Par compatibilité, pour avoir un graphe')
    plt.ylabel('Erreur moyenne')
    
    return 0;

def drawBenchmarkForMultipleValues(y_pred, y_test) :
    
    return 0;

'''
Mettre ici tout les tests
Faire des benchmarks entre les tests, les lancer SUR LE MÊME JEU DE DONNEES !
Implémenter des timers
Potentiellement, faire des recoupements entre les algorithmes, et voir comment
 fusionner les résultats de plusieurs algos pour avoir une prédiction plus fine et exacte.

Tous les alogs devront être fait les uns après les autres, sur le même train_test_split !
'''

csvValuesColumnNumber = 57


# Depuis le csv des mails
csvFilePath = "../spambase/spambase.data";
mailDataset = pd.read_csv(csvFilePath, header = None)  # names=names,

dataFieldsValues = mailDataset.iloc[:, :-1].values  # : signifie "tout" -> :-1 signifie "toutes les colonnes sauf la dernière"
dataLabels = mailDataset.iloc[:, csvValuesColumnNumber].values  

#train_test_split(counts, df['label'], test_size=0.1, random_state=69)  
X_train, X_test, y_train, y_test = train_test_split(dataFieldsValues, dataLabels, test_size=0.2, shuffle=True) # test_size = 1 - train_size

#print(dataFieldsValues)
#print(X_train)

#KNN en exemple
#classifier = KNeighborsClassifier(n_neighbors=4)
#classifier.fit(X_train, y_train)

# Le scaler de KNN ne fonctionne pas avec Naive Bayes car il produit des nombes négatifs,
# ce que Naive Bayes ne supporte pas, malheureusement.
'''
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
'''

classifier = MultinomialNB().fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#print(y_pred)

#print(confusion_matrix(y_test, y_pred))
drawBenchmarkForSingleValue(y_test, y_pred);