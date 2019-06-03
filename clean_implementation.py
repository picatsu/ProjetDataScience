# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28-29

@author: Zylv01

Implémentation regroupant les différents algorithmes, plus propre que de faire pleins de fichiers différents.

Référence super utile pour Naive Bayes :
    https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn/
-> Ce lien est aussi super utile si on veut créer notre propre base de données à partir de mails complets et non formattés.


Il y a 57 champs, et 1 champ pour décrire si le mail est un spam ou non
"""

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import time
import algo.MLP as mlp
import algo.NaiveBayes as nb
# Naive Bayes spécifique


def drawBenchmarkForSingleValue(y_pred, y_test) :
    print(classification_report(y_test, y_pred))
    # Dessin d'un tableau pour pouvoir graphiquement le comparer à KNN
    rangeFirst = [1, 2] # = en python 3,  list(range(1, 3)) #
    # print(rangeFirst)
    # print(y_pred)
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
    
    return 0


def drawBenchmarkForMultipleValues(figureTitle, xLabelName, yLabelName, predictionArrayErrorRatio, predictionArrayName) :
    
    # list(range(minKNumber, maxKNumber)):
    plt.figure(figsize=(14, 6))
    plt.plot(predictionArrayName, predictionArrayErrorRatio, color='red', linestyle='dashed', marker='o',  
             markerfacecolor='blue', markersize=10)
    plt.title(figureTitle)  # 'Taux d\'erreur en fonction de la valeur de K')
    plt.xlabel(xLabelName)  # 'Valeur de K')
    plt.ylabel(yLabelName)  # 'Erreur moyenne')

    return 0


'''
Mettre ici tout les tests
Faire des benchmarks entre les tests, les lancer SUR LE MÊME JEU DE DONNEES !
Implémenter des timers
Potentiellement, faire des recoupements entre les algorithmes, et voir comment
 fusionner les résultats de plusieurs algos pour avoir une prédiction plus fine et exacte.

Tous les alogs devront être fait les uns après les autres, sur le même train_test_split !
'''


def predictWith(algoName, X_train, X_test, y_train, y_test):
    # retourne une liste de valeurs (0 ou 1) : la prédiction, à comparer aux valeurs de la liste y_test
    y_predict = None
    errorOccured = False
    
    # from sklearn.preprocessing import StandardScaler
    
    randSeed = int(time.time() * 10000000000) % 4294967295;  # Modulo la valeur d'un int non signé : 2^32 - 1
    
    print("predictWith randSeed = " + str(randSeed))
    np.random.seed(randSeed)
    
    startTimeMs = int(time.time() * 1000)

    if algoName == "KNN":  # K-Nearest Neighbors
        print("predictWith  " + algoName)
        from sklearn.neighbors import KNeighborsClassifier  # Seulement utiles pour KNN
        """scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)  
        X_test_scaled = scaler.transform(X_test)"""

        classifier = KNeighborsClassifier(n_neighbors=4)  # ♪ avec les 4 voisins les plus proches (stable)
        classifier.fit(X_train, y_train)  # X_train_scaled

        y_predict = classifier.predict(X_test)  # X_test_scaled
    elif algoName == "NaiveBayes":  # NaiveBayes
        y_predict = nb.main(algoName, X_train, X_test, y_train, y_test)
    elif algoName == "MLP": # Backpropagation = MLP pour Multilayer Perceptron
        y_predict = mlp.main(algoName, X_train, X_test, y_train, y_test)
    elif algoName == "LogisticRegression": # LogisticRegression
        print("predictWith  " + algoName)
        from sklearn.linear_model import LogisticRegression

        classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=100000)
        classifier.fit(X_train, y_train)
        # round(LR.score(X,y), 4)  <- retournera presque le même résultat que np.mean(y_predict != y_test)
        # (dans getPredictErrorRatioOf(..))

        y_predict = classifier.predict(X_test)  # X_test_scaled
    elif algoName == "RandomForest": # RandomForest
        print("predictWith  " + algoName)
        from sklearn.ensemble import RandomForestClassifier

        classifier = RandomForestClassifier(n_estimators=100, n_jobs=5)#, max_depth=2, random_state=0)
        classifier.fit(X_train, y_train)
        # round(RF.score(X,y), 4)  <- retournera presque le même résultat que np.mean(y_predict != y_test)
        # (dans getPredictErrorRatioOf(..))

        y_predict = classifier.predict(X_test)  # X_test_scaled
    elif algoName == "Kernel SVM":  # SVM - Support Vector Machine
        print("predictWith  " + algoName)
        from sklearn import svm

        SVM = svm.SVC(decision_function_shape="ovo").fit(X_train, y_train)

        ''' Je sais pas comment faire !
        elif (algoName == "TSNE") : # T-distributed Stochastic Neighbor Embedding
        print("predictWith  " + algoName)
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne.fit(X_train, y_train)

        # = tsne.predict(X_test)
        '''
        y_predict = SVM.predict(X_test)  # X_test_scaled
    else:
        print("ERREUR predictWith : nom de l'algo invalide. algoName = " + algoName)
        y_predict = None
        errorOccured = True
    
    if not errorOccured:
        print(classification_report(y_test, y_predict))

    elapsedTimeMs = int(time.time() * 1000) - startTimeMs
    
    # if (y_predict == None) return None;
    return y_predict, algoName, elapsedTimeMs


def getPredictErrorRatioOf(algoName, X_train, X_test, y_train, y_test) :
    y_predict, algoName2, elapsedTimeMs = predictWith(algoName, X_train, X_test, y_train, y_test)
    localPredictErrorRatio = np.mean(y_predict != y_test)
    
    return localPredictErrorRatio, elapsedTimeMs


def getPredictErrorRatioOfAndAddToLists(algoName, X_train, X_test, y_train, y_test, predictionArrayErrorRatio, predictionArrayName, predictionArrayTimeTookMs) :
    localPredictErrorRatio, elapsedTimeMs = getPredictErrorRatioOf(algoName, X_train, X_test, y_train, y_test)
    
    predictionArrayErrorRatio.append(localPredictErrorRatio)
    predictionArrayName.append(algoName)
    predictionArrayTimeTookMs.append(elapsedTimeMs)

    return 0


def getPredictErrorRatioOfAndAddToLists_withA2List(algoName, a2_X_train, a2_X_test, a2_y_train, a2_y_test, predictionArrayErrorRatio, predictionArrayName, predictionArrayTimeTookMs) :
    iterationNumber = len(a2_X_train)
    
    if iterationNumber <= 0: return
    
    for iIteration in range(0, iterationNumber):
        localPredictErrorRatio, elapsedTimeMs = getPredictErrorRatioOf(algoName, a2_X_train[iIteration], a2_X_test[iIteration], a2_y_train[iIteration], a2_y_test[iIteration])
        
        predictionArrayErrorRatio.append(localPredictErrorRatio)
        predictionArrayName.append(algoName)
        predictionArrayTimeTookMs.append(elapsedTimeMs)

    return 0


# Fonction main
def doFullBenchmark():
    # Chargement initial des données (mails)
    csvValuesColumnNumber = 57
    
    # Depuis le csv des mails
    csvFilePath = "spambase/spambase.data";
    mailDataset = pd.read_csv(csvFilePath, header=None)  # names=names,
    mailDataset.drop(columns=[26,27])  # Drop columns "Georges & 650" contextual false-positives
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

    iterationNumber = 10;

    # Permet d'avoir des jeux de test identiques pour chaque itération
    a2_X_train = []
    a2_X_test = []
    a2_y_train = []
    a2_y_test = []
    
    # Jeu de tests scalé
    a2_X_train_scaled = []
    a2_X_test_scaled = []
    
    from sklearn.preprocessing import StandardScaler
    
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

    # getPredictErrorRatioOfAndAddToLists_withA2List("TSNE", a2_X_train, a2_X_test, a2_y_train, a2_y_test,
    # predictionArrayErrorRatio, predictionArrayName, predictionArrayTimeTookMs)
    getPredictErrorRatioOfAndAddToLists_withA2List("KNN", a2_X_train, a2_X_test, a2_y_train, a2_y_test, predictionArrayErrorRatio, predictionArrayName, predictionArrayTimeTookMs)
    getPredictErrorRatioOfAndAddToLists_withA2List("MLP", a2_X_train, a2_X_test, a2_y_train, a2_y_test, predictionArrayErrorRatio, predictionArrayName, predictionArrayTimeTookMs)
    getPredictErrorRatioOfAndAddToLists_withA2List("NaiveBayes", a2_X_train, a2_X_test, a2_y_train, a2_y_test, predictionArrayErrorRatio, predictionArrayName, predictionArrayTimeTookMs)
    
    getPredictErrorRatioOfAndAddToLists_withA2List("LogisticRegression", a2_X_train, a2_X_test, a2_y_train, a2_y_test, predictionArrayErrorRatio, predictionArrayName, predictionArrayTimeTookMs)
    getPredictErrorRatioOfAndAddToLists_withA2List("RandomForest", a2_X_train, a2_X_test, a2_y_train, a2_y_test, predictionArrayErrorRatio, predictionArrayName, predictionArrayTimeTookMs)
    getPredictErrorRatioOfAndAddToLists_withA2List("Kernel SVM", a2_X_train, a2_X_test, a2_y_train, a2_y_test, predictionArrayErrorRatio, predictionArrayName, predictionArrayTimeTookMs)

    predictionArrayErrorRatioScaled = []  # prédiction, valeurs à comparer à y_test
    predictionArrayNameScaled = []
    predictionArrayTimeTookMsScaled = []
    
    getPredictErrorRatioOfAndAddToLists_withA2List("KNN", a2_X_train_scaled, a2_X_test_scaled, a2_y_train, a2_y_test, predictionArrayErrorRatioScaled, predictionArrayNameScaled, predictionArrayTimeTookMsScaled)
    
    # MLP et NaiveBayes ne supportent pas le scaling
    getPredictErrorRatioOfAndAddToLists_withA2List("MLP", a2_X_train, a2_X_test, a2_y_train, a2_y_test, predictionArrayErrorRatioScaled, predictionArrayNameScaled, predictionArrayTimeTookMsScaled)
    getPredictErrorRatioOfAndAddToLists_withA2List("NaiveBayes", a2_X_train, a2_X_test, a2_y_train, a2_y_test, predictionArrayErrorRatioScaled, predictionArrayNameScaled, predictionArrayTimeTookMsScaled)
    
    getPredictErrorRatioOfAndAddToLists_withA2List("LogisticRegression", a2_X_train_scaled, a2_X_test_scaled, a2_y_train, a2_y_test, predictionArrayErrorRatioScaled, predictionArrayNameScaled, predictionArrayTimeTookMsScaled)
    getPredictErrorRatioOfAndAddToLists_withA2List("RandomForest", a2_X_train_scaled, a2_X_test_scaled, a2_y_train, a2_y_test, predictionArrayErrorRatioScaled, predictionArrayNameScaled, predictionArrayTimeTookMsScaled)
    getPredictErrorRatioOfAndAddToLists_withA2List("Kernel SVM", a2_X_train_scaled, a2_X_test_scaled, a2_y_train, a2_y_test, predictionArrayErrorRatioScaled, predictionArrayNameScaled, predictionArrayTimeTookMsScaled)

    drawBenchmarkForMultipleValues('Non Scalé - Taux d\'erreur en fonction de l\'algo utilisé', 'Algo utilisé',
                                   'Erreur moyenne', predictionArrayErrorRatio, predictionArrayName)
    
    drawBenchmarkForMultipleValues('Scalé - Taux d\'erreur en fonction de l\'algo utilisé', 'Algo utilisé',
                                   'Erreur moyenne', predictionArrayErrorRatioScaled, predictionArrayNameScaled)
    
    drawBenchmarkForMultipleValues("Non Scalé - Temps pris par algorithme", "Algo utilisé", "Temps pris (ms)", predictionArrayTimeTookMs, predictionArrayName)
    drawBenchmarkForMultipleValues("Scalé - Temps pris par algorithme", "Algo utilisé", "Temps pris (ms)", predictionArrayTimeTookMsScaled, predictionArrayNameScaled)

    '''
    for i in range(0, iterationNumber) :
        #X_train, X_test, y_train, y_test = train_test_split(dataFieldsValues, dataLabels, test_size=0.2, shuffle=True) # test_size = 1 - train_size
        
        getPredictErrorRatioOfAndAddToLists_withA2List()
        
        X_train = a2_X_train[i]
        X_test = a2_X_test[i]
        y_train = a2_y_train[i]
        y_test = a2_y_test[i]
        
        getPredictErrorRatioOfAndAddToLists("KNN", X_train, X_test, y_train, y_test, predictionArrayErrorRatio, predictionArrayName)
        getPredictErrorRatioOfAndAddToLists("MLP", X_train, X_test, y_train, y_test, predictionArrayErrorRatio, predictionArrayName)
        getPredictErrorRatioOfAndAddToLists("NaiveBayes", X_train, X_test, y_train, y_test, predictionArrayErrorRatio, predictionArrayName)
    '''

    # print(confusion_matrix(y_test, y_pred))
    # drawBenchmarkForSingleValue(y_test, y_pred);

    return 0


doFullBenchmark()


# Le scaler de KNN ne fonctionne pas avec Naive Bayes car il produit des nombes négatifs,
# ce que Naive Bayes ne supporte pas, malheureusement.
