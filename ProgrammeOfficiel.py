# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 00:11:42 2019

@author: ZazaX.
"""

import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd



def mainMoche() :
    
    csvValuesColumnNumber = 57
    # Depuis le csv des mails
    csvFilePath = "spambase/spambase.data";
    mailDataset = pd.read_csv(csvFilePath, header=None)  # names=names,
    
    dataLabels = mailDataset.iloc[:, csvValuesColumnNumber].values
    mailDataset.drop(columns=[26,27])  # Drop columns "Georges & 650" contextual false-positives
    # Split des colonnes en deux : les valeurs (dataFieldsValues) et le label pour chaque mail (dataLabels)
    # permettant de savoir si c'est un spam (1) ou non
    dataFieldsValues = mailDataset.iloc[:, :-1].values
    
    
    iterationNumber = 20;
    
    X_train, X_test, y_train, y_test = train_test_split(dataFieldsValues, dataLabels, test_size=0.2, shuffle=True) # test_size = 1 - train_size
    
    Tab = []
    
    for iIteration in range(0, iterationNumber) :
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        
        classifier = RandomForestClassifier(n_estimators=80, n_jobs=5)#, max_depth=2, random_state=0)
        classifier.fit(X_train, y_train)
        # round(RF.score(X,y), 4)  <- retournera presque le même résultat que np.mean(y_predict != y_test)
        # (dans getPredictErrorRatioOf(..))
        y_predict = classifier.predict(X_test)  # X_test_scaled
        
        localPredictErrorRatio = np.mean(y_predict != y_test)
        print(1 - localPredictErrorRatio);
        Tab.append(1 - localPredictErrorRatio)
        
    
    
    
    print('#### SCORE RandomForest optimisé  ####')
    print('max : ',max(Tab))
    print('min :',min(Tab))
    print('AVG :',sum(Tab)/len(Tab))
    print('#####################') 
    
    
    
    return


mainMoche()
