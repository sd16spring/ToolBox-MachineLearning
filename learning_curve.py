""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_digits()
print data.DESCR
num_trials = 1000
train_percentages = range(5,95,5)
test_accuracies = numpy.zeros(len(train_percentages))


curve_y = []
model = LogisticRegression(C=10**-10)
for i in train_percentages:
	score = []
	for j in range(num_trials):
		X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, train_size=i)
		model.fit(X_train, Y_train)
		score.append(model.score(X_test, Y_test))
	test_accuracies[i/5-1] = sum(score)/num_trials


fig = plt.figure()
plt.plot(train_percentages, test_accuracies)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.show()
