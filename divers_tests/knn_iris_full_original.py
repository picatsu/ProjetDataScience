# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 15:54:45 2018

@author: Wandrille
"""

import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix


# import some data to play with
iris = datasets.load_iris()


X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=4)  
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test) 

print(y_pred)  # 0 correspond to Versicolor, 1 to Verginica and 2 to Setosa

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))




#error = []
#
## Calculating error for K values between 1 and 40
#for i in range(1, 40):  
#    knn = KNeighborsClassifier(n_neighbors=i)
#    knn.fit(X_train, y_train)
#    pred_i = knn.predict(X_test)
#    error.append(np.mean(pred_i != y_test))
#    
#plt.figure(figsize=(12, 6))  
#plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
#         markerfacecolor='blue', markersize=10)
#plt.title('Error Rate K Value')  
#plt.xlabel('K Value')  
#plt.ylabel('Mean Error')  