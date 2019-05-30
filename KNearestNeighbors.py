# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  # Seulement utiles pour KNN
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import Benchmark
import csv



class KNearestNeighbors:
    
    def __init__(self):
        self.Tab = []
        return 
        
    def run(self):
        iterationNumber = 10
        print("KNN running")
        print("KNearestNeighbors initializing")
        # Chargement initial des données (mails)
        csvValuesColumnNumber = 57
        csvFilePath = "spambase/spambase.data"
        mailDataset = pd.read_csv(csvFilePath, header=None)  # names=names,
        mailDataset.drop(columns=[26, 27])  # Drop columns "Georges & 650" contextual false-positives
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
        # Permet d'avoir des jeux de test identiques pour chaque itération
        a2_X_train = []
        a2_X_test = []
        a2_y_train = []
        a2_y_test = []
        # Jeu de tests scalé
        a2_X_train_scaled = []
        a2_X_test_scaled = []


        for i in range(0, iterationNumber):
            X_train, X_test, y_train, y_test = train_test_split(dataFieldsValues, dataLabels, test_size=0.2, shuffle=True)  # test_size = 1 - train_size
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

        predictionArrayErrorRatio = []  # prédiction, valeurs à comparer à y_test
        predictionArrayName = []
        predictionArrayTimeTookMs = []

        iterationNumber = len(a2_X_train)

        if iterationNumber <= 0:
            return

        for iIteration in range(0, iterationNumber):

            errorOccured = False
            randSeed = int(time.time() * 10000000000) % 4294967295  # Modulo la valeur d'un int non signé : 2^32 - 1

            # print("predictWith randSeed = " + str(randSeed))
            np.random.seed(randSeed)

            startTimeMs = int(time.time() * 1000)

            #print("predictWith  " + "KNN")

            classifier = KNeighborsClassifier(n_neighbors=4)  # ♪ avec les 4 voisins les plus proches (stable)
            classifier.fit(a2_X_train[iIteration], a2_y_train[iIteration])  # X_train_scaled
            y_predict = classifier.predict(a2_X_test[iIteration])  # X_test_scaled
            """
            if not errorOccured:
                print(classification_report(a2_y_test[iIteration], y_predict))
            """

            elapsedTimeMs = int(time.time() * 1000) - startTimeMs

            localPredictErrorRatio = np.mean(y_predict != a2_y_test[iIteration])

            predictionArrayErrorRatio.append(localPredictErrorRatio)
            predictionArrayName.append("KNN")
            predictionArrayTimeTookMs.append(elapsedTimeMs)

        predictionArrayErrorRatioScaled = []  # prédiction, valeurs à comparer à y_test
        predictionArrayNameScaled = []
        predictionArrayTimeTookMsScaled = []

        iterationNumber = len(a2_X_train_scaled)

        if iterationNumber <= 0:
            return

        for iIteration in range(0, iterationNumber):
            errorOccured = False
            randSeed = int(time.time() * 10000000000) % 4294967295  # Modulo la valeur d'un int non signé : 2^32 - 1

            #print("predictWith randSeed = " + str(randSeed))
            np.random.seed(randSeed)

            startTimeMs = int(time.time() * 1000)

            #print("predictWith  " + "KNN")

            classifier = KNeighborsClassifier(n_neighbors=4)  # ♪ avec les 4 voisins les plus proches (stable)
            classifier.fit(a2_X_train_scaled[iIteration], a2_y_train[iIteration])  # X_train_scaled
            y_predict = classifier.predict(a2_X_test_scaled[iIteration])  # X_test_scaled
            """if not errorOccured:
                #print(classification_report(a2_y_test[iIteration], y_predict))
                print()
            """

            elapsedTimeMs = int(time.time() * 1000) - startTimeMs
            self.Tab.append(1 - localPredictErrorRatio)
            

            localPredictErrorRatio = np.mean(y_predict != a2_y_test[iIteration])
            

            predictionArrayErrorRatioScaled.append(localPredictErrorRatio)
            predictionArrayNameScaled.append("KNN")
            predictionArrayTimeTookMsScaled.append(elapsedTimeMs)
            """
        Benchmark.drawBenchmarkForMultipleValues('Non Scalé - Taux d\'erreur en fonction de l\'algo utilisé', 'Algo utilisé', 'Erreur moyenne', predictionArrayErrorRatio, predictionArrayName)
        Benchmark.drawBenchmarkForMultipleValues('Scalé - Taux d\'erreur en fonction de l\'algo utilisé', 'Algo utilisé', 'Erreur moyenne', predictionArrayErrorRatioScaled, predictionArrayNameScaled)
        Benchmark.drawBenchmarkForMultipleValues("Non Scalé - Temps pris par algorithme", "Algo utilisé", "Temps pris (ms)", predictionArrayTimeTookMs, predictionArrayName)
        Benchmark.drawBenchmarkForMultipleValues("Scalé - Temps pris par algorithme", "Algo utilisé", "Temps pris (ms)", predictionArrayTimeTookMsScaled, predictionArrayNameScaled)
        """
        #draw(predictionArrayErrorRatio, predictionArrayName, predictionArrayErrorRatioScaled, predictionArrayNameScaled, predictionArrayTimeTookMsScaled, predictionArrayTimeTookMs )
        return self.Tab



def test():
    Tab = KNearestNeighbors().run()
    print('####### SCORE KNN ####')
    print('max : ',max(Tab))
    print('min :',min(Tab))
    print('AVG :',sum(Tab)/len(Tab))
    print('#####################')
def draw(predictionArrayErrorRatio, predictionArrayName, predictionArrayErrorRatioScaled, predictionArrayNameScaled, predictionArrayTimeTookMsScaled, predictionArrayTimeTookMs ):
    Benchmark.drawBenchmarkForMultipleValues('Non Scalé - Taux d\'erreur en fonction de l\'algo utilisé', 'Algo utilisé', 'Erreur moyenne', predictionArrayErrorRatio, predictionArrayName)
    Benchmark.drawBenchmarkForMultipleValues('Scalé - Taux d\'erreur en fonction de l\'algo utilisé', 'Algo utilisé', 'Erreur moyenne', predictionArrayErrorRatioScaled, predictionArrayNameScaled)
    Benchmark.drawBenchmarkForMultipleValues("Non Scalé - Temps pris par algorithme", "Algo utilisé", "Temps pris (ms)", predictionArrayTimeTookMs, predictionArrayName)
    Benchmark.drawBenchmarkForMultipleValues("Scalé - Temps pris par algorithme", "Algo utilisé", "Temps pris (ms)", predictionArrayTimeTookMsScaled, predictionArrayNameScaled)

def somme(liste):
    _somme = 0
    for i in liste:
        _somme = _somme + i
    return _somme

def moyenne(liste):
    return somme(liste)/len(liste)
    
    
    

    
