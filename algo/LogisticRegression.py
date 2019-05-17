def main(algoName, X_train, X_test, y_train, y_test):
    print("predictWith  " + algoName)
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=100000)
    classifier.fit(X_train, y_train)
    # round(LR.score(X,y), 4)  <- retournera presque le même résultat que np.mean(y_predict != y_test)
    # (dans getPredictErrorRatioOf(..))

    return classifier.predict(X_test)  # X_test_scaled