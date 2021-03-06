# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import Benchmark


class RandomForest:
    def __init__(self):
        print("RandomForest initializing")
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
        iterationNumber = 2
        # Permet d'avoir des jeux de test identiques pour chaque itération
        a2_X_train = []
        a2_X_test = []
        a2_y_train = []
        a2_y_test = []
        # Jeu de tests scalé
        a2_X_train_scaled = []
        a2_X_test_scaled = []

        from sklearn.preprocessing import StandardScaler

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

            print("predictWith randSeed = " + str(randSeed))
            np.random.seed(randSeed)

            startTimeMs = int(time.time() * 1000)

            print("predictWith  " + "RF")
            from sklearn.ensemble import RandomForestClassifier

            classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
            classifier.fit(a2_X_train[iIteration], a2_y_train[iIteration])  # X_train_scaled
            y_predict = classifier.predict(a2_X_test[iIteration])  # X_test_scaled

            if not errorOccured:
                print(classification_report(a2_y_test[iIteration], y_predict))


            elapsedTimeMs = int(time.time() * 1000) - startTimeMs

            localPredictErrorRatio = np.mean(y_predict != a2_y_test[iIteration])

            predictionArrayErrorRatio.append(localPredictErrorRatio)
            predictionArrayName.append("RF")
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

            print("predictWith randSeed = " + str(randSeed))
            np.random.seed(randSeed)

            startTimeMs = int(time.time() * 1000)

            print("predictWith  " + "RF")
            from sklearn.ensemble import RandomForestClassifier

            classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
            classifier.fit(a2_X_train_scaled[iIteration], a2_y_train[iIteration])  # X_train_scaled
            y_predict = classifier.predict(a2_X_test_scaled[iIteration])  # X_test_scaled

            if not errorOccured:
                print(classification_report(a2_y_test[iIteration], y_predict))

            elapsedTimeMs = int(time.time() * 1000) - startTimeMs

            localPredictErrorRatio = np.mean(y_predict != a2_y_test[iIteration])

            predictionArrayErrorRatioScaled.append(localPredictErrorRatio)
            predictionArrayNameScaled.append("RF")
            predictionArrayTimeTookMsScaled.append(elapsedTimeMs)

        Benchmark.drawBenchmarkForMultipleValues('Non Scalé - Taux d\'erreur en fonction de l\'algo utilisé', 'Algo utilisé', 'Erreur moyenne', predictionArrayErrorRatio, predictionArrayName)
        Benchmark.drawBenchmarkForMultipleValues('Scalé - Taux d\'erreur en fonction de l\'algo utilisé', 'Algo utilisé', 'Erreur moyenne', predictionArrayErrorRatioScaled, predictionArrayNameScaled)
        Benchmark.drawBenchmarkForMultipleValues("Non Scalé - Temps pris par algorithme", "Algo utilisé", "Temps pris (ms)", predictionArrayTimeTookMs, predictionArrayName)
        Benchmark.drawBenchmarkForMultipleValues("Scalé - Temps pris par algorithme", "Algo utilisé", "Temps pris (ms)", predictionArrayTimeTookMsScaled, predictionArrayNameScaled)

    def run(self):
        print("RF running")


def test():
    test = RandomForest()
    test.run()
