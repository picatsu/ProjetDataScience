from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import time
import pandas as pd

tmps1=time.time()

class RandomForest:
    def __init__(self):
        self.lR = RandomForestClassifier(n_jobs = -1, random_state=9,oob_score = False, bootstrap=True)
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
        accuracy_RF = accuracy_score(self.Y_test, predict_lR)
        precision_RF = precision_score(self.Y_test, predict_lR)
        recall_RF = recall_score(self.Y_test, predict_lR)
        f1_score_RF = f1_score(self.Y_test, predict_lR)
        auc_RF = roc_auc_score(self.Y_test, predict_lR)
        print('accuracy ', accuracy_RF)
        print('precision ', precision_RF)
        print('recall ', recall_RF)
        print('f1_score ', f1_score_RF)
        print('auc ', auc_RF)
        print("************************* Random forest Results *****************************")
        print("rapport de classification :")
        print(metrics.classification_report(self.Y_test, predict_lR))
        print("score de précision :")
        print(metrics.accuracy_score(self.Y_test, predict_lR))
        return metrics.accuracy_score(self.Y_test, predict_lR)

tmps2=time.time()-tmps1
print("Temps d'execution = ", tmps2)
