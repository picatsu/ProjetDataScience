from numpy.core.tests.test_mem_overlap import xrange
from sklearn import preprocessing, metrics
import numpy as np
import pprint
from sklearn.metrics import confusion_matrix

NB_TRAIN = (int)(4600 * 0.8)


class Backpropagation:

    def __init__(self):
        self.file_data = r'./base/spambase.data'
        self.X = []
        self.Y = []

    def derivative(self, x):
        return x * (1.0 - x)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def result(self):

        with open(self.file_data) as f:
            for line in f:
                curr = line.split(',')
                new_curr = [1]
                for item in curr[:len(curr) - 1]:
                    new_curr.append(float(item))
                self.X.append(new_curr)
                self.Y.append([float(curr[-1])])

        self.X = np.array(self.X)
        self.X = preprocessing.scale(self.X)  # feature scaling
        self.Y = np.array(self.Y)

        # the first 2500 out of 3000 emails will serve as training data
        X_train = self.X[0:NB_TRAIN]
        Y_train = self.Y[0:NB_TRAIN]
        # the rest 500 emails will serve as testing data
        X_test = self.X[NB_TRAIN:]
        Y_test = self.Y[NB_TRAIN:]

        self.X = X_train
        self.Y = Y_train


        dim1 = len(X_train[0])
        dim2 = 4
        # randomly initialize the weight vectors
        np.random.seed(1)
        weight0 = 2 * np.random.random((dim1, dim2)) - 1
        weight1 = 2 * np.random.random((dim2, 1)) - 1

        # you can change the number of iterations
        for j in xrange(20000):
            # first evaluate the output for each training email
            layer_0 = X_train
            layer_1 = self.sigmoid(np.dot(layer_0, weight0))
            layer_2 = self.sigmoid(np.dot(layer_1, weight1))
            # calculate the error
            layer_2_error = Y_train - layer_2
            # perform back propagation
            layer_2_delta = layer_2_error * self.derivative(layer_2)
            layer_1_error = layer_2_delta.dot(weight1.T)
            layer_1_delta = layer_1_error * self.derivative(layer_1)
            # update the weight vectors
            weight1 += layer_1.T.dot(layer_2_delta)
            weight0 += layer_0.T.dot(layer_1_delta)

        # evaluation on the testing data
        layer_0 = X_test
        layer_1 = self.sigmoid(np.dot(layer_0, weight0))
        layer_2 = self.sigmoid(np.dot(layer_1, weight1))

        correct = 0
        # if the output is > 0.5, then label as spam else no spam
        for i in xrange(len(layer_2)):
            if (layer_2[i][0] > 0.5):
                layer_2[i][0] = 1
            else:
                layer_2[i][0] = 0
            if (layer_2[i][0] == Y_test[i][0]):
                correct += 1

        total = len(layer_2)
        accuracy = correct / len(layer_2)
        print("total = ", total)
        print("correct =", correct)
        print("accuracy = ", accuracy)

        print("************************* Backpropagation Results *****************************")
        print("rapport de classification :")
        print(metrics.classification_report(Y_test, layer_2))
        print("score de précision :")
        print(metrics.accuracy_score(Y_test, layer_2))
        return accuracy
