def main(algoName, X_train, X_test, y_train, y_test):
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

    return SVM.predict(X_test)  # X_test_scaled