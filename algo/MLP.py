def main(algoName, X_train, X_test, y_train, y_test):
    print("predictWith  " + algoName)
    # Import à chaque fois pour réinitialiser les tests !
    from sklearn.neural_network import MLPClassifier

    classifier = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  # mlp -> classifier
    classifier.fit(X_train, y_train.ravel())

    return classifier.predict(X_test)  # X_test_scaled