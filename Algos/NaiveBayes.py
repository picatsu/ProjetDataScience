# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:04:51 2019

@author: m6d
"""

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB


class NaiveBayes:
    def __init__(self, algoName):
        self.algoName = algoName
        print(algoName)
        # Import à chaque fois pour réinitialiser les tests !
      # Chargement initial des données (mails)
        csvValuesColumnNumber = 57
      # Depuis le csv des mails
        csvFilePath = "spambase/spambase.data";
        mailDataset = pd.read_csv(csvFilePath, header = None)  # names=names,
      # Split des colonnes en deux : les valeurs (dataFieldsValues) et le label pour chaque mail (dataLabels)
      # permettant de savoir si c'est un spam (1) ou non
        dataFieldsValues = mailDataset.iloc[:, :-1].values
      # : signifie "tout" -> :-1 signifie "toutes les colonnes sauf la dernière"
        dataLabels = mailDataset.iloc[:, csvValuesColumnNumber].values
    # train_test_split(counts, df['label'], test_size=0.1, random_state=69)
    
    # Split des lignes de spambase et shuffle pour avoir un échantillon aléatoire
    # X_train : valeurs d'entraînement
    # y_train : labels d'entraînement (associés à chaque valeur)
    # X_test : valeurs pour le test
    # y_test : labels pour vérifier le test
        iterationNumber = 2;
      # Permet d'avoir des jeux de test identiques pour chaque itération
        a2_X_train = []
        a2_X_test = []
        a2_y_train = []
        a2_y_test = []
      # Jeu de tests scalé
        a2_X_train_scaled = []
        a2_X_test_scaled = []
        for i in range(0, iterationNumber) :
            X_train, X_test, y_train, y_test = train_test_split(dataFieldsValues, dataLabels, test_size=0.2, shuffle=True) # test_size = 1 - train_size
            a2_X_train.append(X_train)
            a2_X_test.append(X_test)
            a2_y_train.append(y_train)
            a2_y_test.append(y_test)
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)  
            X_test_scaled = scaler.transform(X_test)
            a2_X_train_scaled.append(X_train_scaled)
            a2_X_test_scaled.append(X_test_scaled)
        predictionArrayErrorRatio = [] # prédiction, valeurs à comparer à y_test
        predictionArrayName = []
        predictionArrayTimeTookMs = []
      
      
    def run(self):
        y_predict = None
        errorOccured = False
        # from sklearn.preprocessing import StandardScaler
        randSeed = int(time.time() * 10000000000) % 4294967295;  # Modulo la valeur d'un int non signé : 2^32 - 1
        print("predictWith randSeed = " + str(randSeed))
        np.random.seed(randSeed)
        startTimeMs = int(time.time() * 1000)
        classifier = MultinomialNB();
        classifier.fit(X_train, y_train)
        """
        if not errorOccured:
            print(classification_report(y_test, y_predict))
        """
        elapsedTimeMs = int(time.time() * 1000) - startTimeMs
        # if (y_predict == None) return None;
        return y_predict, elapsedTimeMs
    
algo1 = NaiveBayes("NaiveBayes")
algo1.run()
