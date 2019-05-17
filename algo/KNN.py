def main(algoName, X_train, X_test, y_train, y_test):
    print("predictWith  " + algoName)
    from sklearn.neighbors import KNeighborsClassifier  # Seulement utiles pour KNN
    """scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)  
    X_test_scaled = scaler.transform(X_test)"""

    classifier = KNeighborsClassifier(n_neighbors=4)  # ♪ avec les 4 voisins les plus proches (stable)
    classifier.fit(X_train, y_train)  # X_train_scaled

    return classifier.predict(X_test)  # X_test_scaled