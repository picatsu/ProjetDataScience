
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import time


class LogisticRegressionAlgo:

    def __init__(self):
        self.lR = LogisticRegression()

        # Import à chaque fois pour réinitialiser les tests !
        # Chargement initial des données (mails)
        csvValuesColumnNumber = 57
        # Depuis le csv des mails
        csvFilePath = "./base/spambase.data";
        mailDataset = pd.read_csv(csvFilePath, header=None)  # names=names,
        # Split des colonnes en deux : les valeurs (dataFieldsValues) et le label pour chaque mail (dataLabels)
        # permettant de savoir si c'est un spam (1) ou non
        dataFieldsValues = mailDataset.iloc[:, :-1].values
        # : signifie "tout" -> :-1 signifie "toutes les colonnes sauf la dernière"
        dataLabels = mailDataset.iloc[:, csvValuesColumnNumber].values
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(dataFieldsValues, dataLabels, test_size=0.2,shuffle=True)



    def result(self):
        self.lR.fit(self.X_train, self.Y_train)
        predict_lR = self.lR.predict(self.X_test)

        print("************************* Logistic Regression Results *****************************")
        print("rapport de classification :")
        print(metrics.classification_report(self.Y_test, predict_lR))
        print("score de précision :")
        print(metrics.accuracy_score(self.Y_test, predict_lR))
        return metrics.accuracy_score(self.Y_test, predict_lR)

        return metrics.accuracy_score(self.Y_test, predict_lR)

