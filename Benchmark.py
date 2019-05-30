import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def drawBenchmarkForSingleValue(y_pred, y_test):
    print(classification_report(y_test, y_pred))
    # Dessin d'un tableau pour pouvoir graphiquement le comparer a KNN
    rangeFirst = [1, 2] # = en python 3,  list(range(1, 3)) #
    # print(rangeFirst)
    # print(y_pred)
    predictedRatio = np.mean(y_pred != y_test)
    error = [predictedRatio, predictedRatio  ]  # [3.542, 8.612]

    '''error = []
    error.append(predictedRatio)
    error.append(predictedRatio)'''

    print('ICI PREDICT RATIO')
    print(predictedRatio)

    plt.figure(figsize=(14, 6))
    plt.plot(rangeFirst, error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Taux d\'erreur pour Naive Bayes')
    plt.xlabel('Par compatibilite, pour avoir un graphe')
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
Faire des benchmarks entre les tests, les lancer SUR LE MEME JEU DE DONNEES !
Implementer des timers
Potentiellement, faire des recoupements entre les algorithmes, et voir comment
 fusionner les resultats de plusieurs algos pour avoir une prediction plus fine et exacte.

Tous les alogs devront etre fait les uns apres les autres, sur le meme train_test_split !
'''


def getPredictErrorRatioOf(X_train, X_test, y_train, y_test) :
    y_predict, algoName2, elapsedTimeMs = predictWith(algoName, X_train, X_test, y_train, y_test)
    localPredictErrorRatio = np.mean(y_predict != y_test)

    return localPredictErrorRatio, elapsedTimeMs


def getPredictErrorRatioOfAndAddToLists(X_train, X_test, y_train, y_test, predictionArrayErrorRatio, predictionArrayName, predictionArrayTimeTookMs) :
    localPredictErrorRatio, elapsedTimeMs = getPredictErrorRatioOf(algoName, X_train, X_test, y_train, y_test)

    predictionArrayErrorRatio.append(localPredictErrorRatio)
    predictionArrayName.append(algoName)
    predictionArrayTimeTookMs.append(elapsedTimeMs)

    return 0


def getPredictErrorRatioOfAndAddToLists_withA2List(a2_X_train, a2_X_test, a2_y_train, a2_y_test, predictionArrayErrorRatio, predictionArrayName, predictionArrayTimeTookMs) :
    iterationNumber = len(a2_X_train)

    if iterationNumber <= 0: return

    for iIteration in range(0, iterationNumber):
        localPredictErrorRatio, elapsedTimeMs = getPredictErrorRatioOf(algoName, a2_X_train[iIteration], a2_X_test[iIteration], a2_y_train[iIteration], a2_y_test[iIteration])

        predictionArrayErrorRatio.append(localPredictErrorRatio)
        predictionArrayName.append(algoName)
        predictionArrayTimeTookMs.append(elapsedTimeMs)

    return 0
