def main(algoName, X_train, X_test, y_train, y_test):
    print("predictWith  " + algoName)
    from sklearn.ensemble import RandomForestClassifier

    classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    classifier.fit(X_train, y_train)
    # round(RF.score(X,y), 4)  <- retournera presque le même résultat que np.mean(y_predict != y_test)
    # (dans getPredictErrorRatioOf(..))

    return classifier.predict(X_test)  # X_test_scaled
