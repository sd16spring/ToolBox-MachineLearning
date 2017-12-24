""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

def display_nums(data, ):
	digits = load_digits()
	print digits.DESCR
	fig = plt.figure()
	for i in range(10):
		subplot = fig.add_subplot(5,2,i+1)
		subplot.matshow(numpy.reshape(digits.data[i],(8,8)),cmap='gray')
	plt.show()

def mlr(data, num_trials, train_percentages, c=10**-10):
	test_accuracies = [] 
	for percentage in train_percentages:
		x = 0
		new_list = []
		while x < num_trials:
			x += 1
			X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=percentage)
			model = LogisticRegression(C=c)
			model.fit(X_train, y_train)
			new_list.append(model.score(X_test,y_test))
		average = sum(new_list)/len(new_list)
		test_accuracies.append(average)
	return test_accuracies

def learningcurve(data, num_trials, train_percentages, c=10**-10):
	test_accuracies = mlr(data, num_trials, train_percentages, c)
	print train_percentages
	print test_accuracies
	fig = plt.figure()
	plt.plot(train_percentages, test_accuracies)
	plt.xlabel('Percentage of Data Used for Training')
	plt.ylabel('Accuracy on Test Set')
	plt.suptitle('Regression With c ='+str(c))
	plt.show()

if __name__ == "__main__":
	data = load_digits()
	num_trials = 50
	train_percentages = range(5,95,5)
	learningcurve(data, num_trials, train_percentages)