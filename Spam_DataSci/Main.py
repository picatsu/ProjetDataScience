


from Algo.Backpropagation import Backpropagation
from Algo.LogisticRegression import LogisticRegressionAlgo
from Algo.RandomForest import RandomForest


import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd




def main():


    print("############Backpropagation############")
    backpropagation = Backpropagation()
    backpropagation_acc = backpropagation.result()
    print("-------------------------------------------")
    print("")

    print("############LogisticRegression############")
    lRalgo = LogisticRegressionAlgo()
    logregression_acc = lRalgo.result()
    print("-------------------------------------------")
    print("")

    print("############RandomForest############")
    randomForest = RandomForest()
    randomforest_acc = randomForest.result()
    print("-------------------------------------------")
    print("")

    accuracy = [backpropagation_acc, logregression_acc, randomforest_acc]
    accuracy_list = pd.Series(accuracy)
    accuracy_labels = ["Backpropagation", "LogisticRegression", "RandomForest"]

    plt.figure(figsize=(12, 8))
    ax = accuracy_list.plot(kind='bar')
    ax.set_title('Histogramme des algorithmes utilisés')
    ax.set_xlabel('Nom')
    ax.set_ylabel('Précision')
    ax.set_xticklabels(accuracy_labels)

    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    rects = ax.patches


    for rect in rects:
         # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.2f}".format(y_value)

        # Create annotation
        plt.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.

    plt.show()


main()