""" Exploring learning curves for classification of handwritten digits """

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

# train a model with training percentages between 5 and 90 (see train_percentages) and evaluate
# the resultant accuracy.
# You should repeat each training percentage num_trials times to smooth out variability
# for consistency with the previous example use model = LogisticRegression(C=10**-10) for your learner

def train_model(train_percentage):
	train_size = train_percentage / 100.0

	X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=train_size)
	model = LogisticRegression(C=10**-14)
	model.fit(X_train, y_train)

	return model.score(X_test,y_test)

for index in range(len(train_percentages)):
	print index
	temp = []
	for i in range(50):
		temp.append(train_model(train_percentages[index]))
	test_accuracies[index] = numpy.mean(temp)


fig = plt.figure()
plt.plot(train_percentages, test_accuracies)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.show()


'''CODE GRAVEYARD'''
#inside train_model:
	# print "Train accuracy %f" %model.score(X_train,y_train)
	# print "Test accuracy %f"%model.score(X_test,y_test)
	