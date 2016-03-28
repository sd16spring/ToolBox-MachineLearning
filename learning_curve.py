""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_digits()
#print data.DESCR
num_trials = 100

train_percentages = range(5,95,5)
test_accuracies = numpy.zeros(len(train_percentages))

def trainer(percent, num_trials):
    results = []
    model = LogisticRegression(C=10**-4)
    for i in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=(percent/float(100)))
        model.fit(X_train, y_train)
        results.append(model.score(X_test,y_test))
    return sum(results)/float(num_trials)

results = [trainer(percent, num_trials) for percent in train_percentages]
fig = plt.figure()
plt.plot(train_percentages, results)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.show()
