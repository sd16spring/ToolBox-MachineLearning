""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_digits()
num_trials = 100
train_percentages = range(5,90,5)
test_accuracies = [] #numpy.zeros(len(train_percentages))

for i in train_percentages: #runs through different percentages
	list_of_n = []
	for n in range(num_trials): #runs a given number of times --> average data
		#Partition data into two sets: training and testing.
		X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=i)
		model = LogisticRegression(C=10**-10)#10**-10) #use the Multinomial Logistic Regression algorithm
		model.fit(X_train, y_train)
		accuracy_of_n = model.score(X_test,y_test) #find accuracy
		list_of_n.append(accuracy_of_n) #add each new score to list
	average_n = sum(list_of_n)/len(list_of_n) #average list
	test_accuracies.append(average_n) #append average to test_accuracies list

fig = plt.figure()
plt.plot(train_percentages, test_accuracies)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.show()
