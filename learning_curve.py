""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_digits()
num_trials = 100
train_percentages = range(5,95,5)
test_accuracies = [] # numpy.zeros(len(train_percentages))

for n in train_percentages: # For each number within train_percentages
    average_test = 0
    for i in range(0,num_trials): # Run each percentage num_trials times
       X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size = n)
       model = LogisticRegression(C=10**-10)
       model.fit(X_train, y_train)
       average_test += model.score(X_test, y_test) # Take average of results
    test_accuracies.append(average_test/num_trials) # append average accuracy to test_accuracies

fig = plt.figure()
plt.plot(train_percentages, test_accuracies)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.show()