""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

def display_nums(digits):
	fig = plt.figure()
	for i in range(10):
		subplot = fig.add_subplot(5,2,i+1)
		subplot.matshow(numpy.reshape(digits.data[i],(8,8)),cmap='gray')
	plt.show()

def mlr(digits, num_trials, train_percentages, c=10**-10):
	test_accuracies = [] 
	for percentage in train_percentages:
		x = 0
		new_list = []
		while x < num_trials:
			x += 1
			X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=percentage)
			model = LogisticRegression(C=c)
			model.fit(X_train, y_train)
			new_list.append(model.score(X_test,y_test))
		average = sum(new_list)/len(new_list)
		test_accuracies.append(average)
	return test_accuracies

def learning_curve(train_percentages, test_accuracies):
	print train_percentages
	print test_accuracies
	fig = plt.figure()
	plt.plot(train_percentages, test_accuracies)
	plt.xlabel('Percentage of Data Used for Training')
	plt.ylabel('Accuracy on Test Set')
	plt.suptitle('Regression Result')
	plt.show()

def cross_validation(digits):
	tuned_parameters = [{'C': [10**-4, 10**-2, 10**0, 10**2, 10**4]}]
	X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=.9)

	model = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5)
	model.fit(X_train, y_train)

	print "best estimator: \n", model.best_estimator_
	print "score: ", model.score(X_test, y_test)


if __name__ == "__main__":
	# Load the digits dataset from scikit-learn
	digits = load_digits()
	# display_nums(digits)
	num_trials = 50
	train_percentages = range(5,95,5)
	c = 10**-1
	cross_validation(digits)