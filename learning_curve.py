""" Exploring learning curves for classification of handwritten digits """

"""
Completed by Kevin Zhang

Sofware Design Spring 2016
"""

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_digits()
print data.DESCR
num_trials = 100
train_percentages = range(5,95,5)
test_accuracies = numpy.zeros(len(train_percentages))

# train a model with training percentages between 5 and 90 (see train_percentages) and evaluate
# the resultant accuracy.
# You should repeat each training percentage num_trials times to smooth out variability
# for consistency with the previous example use model = LogisticRegression(C=10**-10) for your learner



for training_index in range(len(train_percentages)):

	data = load_digits()
	stablizing_value = 0;   #temp variable to hold a bunch of value for smoothing out variability

	for i in range(num_trials):    #repeated the test for each train_size value 10 times for more stability and smoothness
		X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=train_percentages[training_index]/100.0)
		model = LogisticRegression(C=10**-10)
		model.fit(X_train, y_train)
		print "Training with {}%".format(train_percentages[training_index])
		print "Train accuracy %f" %model.score(X_train,y_train)
		print "Test accuracy %f"%model.score(X_test,y_test)
		print ''
		stablizing_value +=model.score(X_test, y_test)

	stablizing_value /= num_trials	  #take the average of all the values you accumulated
	test_accuracies[training_index] = stablizing_value


fig = plt.figure()
plt.plot(train_percentages, test_accuracies)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.show()
