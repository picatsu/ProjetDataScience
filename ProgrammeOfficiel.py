# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 00:11:42 2019

@author: ZazaX.
"""

import numpy as np
from sklearn.model_selection import train_test_split 
#from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import time


def mainMoche() :
    
    csvValuesColumnNumber = 57
    # Depuis le csv des mails
    csvFilePath = "spambase/spambase.data";
    csvFilePath_WANDRILLE = "spambase/spambase.data" #_de_Wandrille
    '''
    Pour Félix : plus rien à faire ! C'est fonctionnel :)
    
    TODO : changer le fichier csvFilePath_WANDRILLE de nom
           et mettre benchmark_wandrille à True
           
           et le tour est joué !
    '''
    
    
    benchmark_wandrille = False;
    
    
    
    if (benchmark_wandrille) :
        mailDataset_WANDRILLE = pd.read_csv(csvFilePath_WANDRILLE, header=None)  # names=names,
        dataLabels_WANDRILLE = mailDataset_WANDRILLE.iloc[:, csvValuesColumnNumber].values
        mailDataset_WANDRILLE.drop(columns=[26,27])  # Drop columns "Georges & 650" contextual false-positives
        # Split des colonnes en deux : les valeurs (dataFieldsValues) et le label pour chaque mail (dataLabels)
        # permettant de savoir si c'est un spam (1) ou non
        dataFieldsValues_WANDRILLE = mailDataset_WANDRILLE.iloc[:, :-1].values
    
    
    
    
    mailDataset = pd.read_csv(csvFilePath, header=None)  # names=names,
    
    dataLabels = mailDataset.iloc[:, csvValuesColumnNumber].values
    mailDataset.drop(columns=[26,27])  # Drop columns "Georges & 650" contextual false-positives
    # Split des colonnes en deux : les valeurs (dataFieldsValues) et le label pour chaque mail (dataLabels)
    # permettant de savoir si c'est un spam (1) ou non
    dataFieldsValues = mailDataset.iloc[:, :-1].values
    
    
    iterationNumber = 20;
    
    X_train, X_test, y_train, y_test = train_test_split(dataFieldsValues, dataLabels, test_size=0.2, shuffle=True) # test_size = 1 - train_size
    
    Tab = []
    Tab_Wandrille = []
    
    for iIteration in range(0, iterationNumber) :
        
        # inutile avec cette version de l'algo : from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        
        classifier = RandomForestClassifier(n_estimators=80, n_jobs=5)#, max_depth=2, random_state=0)
        classifier.fit(X_train, y_train)
        # round(RF.score(X,y), 4)  <- retournera presque le même résultat que np.mean(y_predict != y_test)
        # (dans getPredictErrorRatioOf(..))
        
        
        startTimeMs = int(time.time() * 1000)
        
        if (benchmark_wandrille) :
            y_predict_Wandrille = classifier.predict(dataFieldsValues_WANDRILLE)  # X_test_scaled
            localPredictErrorRatio_Wandrille = np.mean(y_predict_Wandrille != dataLabels_WANDRILLE)
            elapsedTimeMs = int(time.time() * 1000) - startTimeMs
            print("Wandrille : " + str(1 - localPredictErrorRatio_Wandrille) + "  en " + str(elapsedTimeMs) + " ms.");
            Tab_Wandrille.append(1 - localPredictErrorRatio_Wandrille)
        else :
            y_predict = classifier.predict(X_test)  # X_test_scaled
            localPredictErrorRatio = np.mean(y_predict != y_test)
            elapsedTimeMs = int(time.time() * 1000) - startTimeMs
            print("localTest : " + str(1 - localPredictErrorRatio) + "  en " + str(elapsedTimeMs) + " ms.");
            Tab.append(1 - localPredictErrorRatio)
        
        
        
        
        
    
    if (benchmark_wandrille) :
        print('#### SCORE RandomForest WANDRILLE  ####')
        print('max : ',max(Tab_Wandrille))
        print('min :',min(Tab_Wandrille))
        print('AVG :',sum(Tab_Wandrille)/len(Tab_Wandrille))
        print('#####################') 
    else :
        print('#### SCORE RandomForest optimisé  ####')
        print('max : ',max(Tab))
        print('min :',min(Tab))
        print('AVG :',sum(Tab)/len(Tab))
        print('#####################') 
        print('') 
        print('') 
    
    
    return


mainMoche()
