#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

def main():
    df = pd.read_csv("delay_time.csv")
    X = df[['mean','stdev']]
    Y = df['label'].map({'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5})
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) # dividing the data to 80% as training data, 20% as testing data

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X, Y)
    Y_pred = lr.predict(X_test)
    C = confusion_matrix(Y_test, Y_pred)
    # Normalization
    NC = C / C.astype(np.float).sum(axis=1)
    print(NC)
    for r in NC:
        for c in r:
            print("{}".format(c), end=",")
    # plot
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((Y_train, Y_test))

    plot_decision_regions(X_combined, y_combined, clf=lr)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-150, 50)
    plt.ylim(0, 70)
    plt.legend(loc='upper left')
    plt.savefig("lr.png")
    plt.close('all')

if __name__ == '__main__':
    main()
