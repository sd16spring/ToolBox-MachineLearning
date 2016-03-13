""" Exploring learning curves for classification of handwritten digits."""

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


data = load_digits()
print data.DESCR
num_trials = 10
train_percentages = range(5,95,5)
test_accuracies = numpy.zeros(len(train_percentages))


for i in range(len(train_percentages)):

    training = train_percentages[i]/100.0
    results = []

    for j in range(num_trials):

        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size= training)
        model = LogisticRegression(C=15**-10)
        model.fit(X_train, y_train)
        score = model.score(X_test,y_test)

        results.append(score)

    test_accuracies[i] = numpy.average(results)


fig = plt.figure()
plt.plot(train_percentages, test_accuracies)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.show()
