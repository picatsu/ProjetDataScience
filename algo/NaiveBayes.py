def main(algoName, X_train, X_test, y_train, y_test):
    print("predictWith  " + algoName)
    # Import à chaque fois pour réinitialiser les tests !
    from sklearn.naive_bayes import MultinomialNB

    classifier = MultinomialNB();
    classifier.fit(X_train, y_train)

    return classifier.predict(X_test)  # X_test_scaled