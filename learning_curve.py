""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
#prewritten code:
data = load_digits()
print data.DESCR
num_trials = 50
train_percentages = range(5,95,5)
test_accuracies = [] #numpy.zeros(len(train_percentages)) #couldn't find a way to change these zero values

# train a model with training percentages between 5 and 90 (see train_percentages) and evaluate
# the resultant accuracy.
# You should repeat each training percentage num_trials times to smooth out variability
# for consistency with the previous example use model = LogisticRegression(C=10**-10) for your learner


for percentage in train_percentages:
	x = 0
	new_list = []
	while x < num_trials:
		x += 1
		X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=percentage)
		model = LogisticRegression(C=100)
		model.fit(X_train, y_train)
		new_list.append(model.score(X_test,y_test))
		#print "Train accuracy %f" %model.score(X_train,y_train)
		#print "Test accuracy %f"%model.score(X_test,y_test) 
	average = sum(new_list)/len(new_list)
	test_accuracies.append(average)


print train_percentages
print test_accuracies
fig = plt.figure()
plt.plot(train_percentages, test_accuracies)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.show()







# #from the first section, loading digits and displaying 10 examples:
# digits = load_digits()
# print digits.DESCR
# fig = plt.figure()
# for i in range(10):
#     subplot = fig.add_subplot(5,2,i+1)
#     subplot.matshow(numpy.reshape(digits.data[i],(8,8)),cmap='gray')


# #from the second section:
# data = load_digits()
# #splitting the data into two sets?
# X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=0.3)
# #training on one set
# model = LogisticRegression(C=10**-10)
# #testing on the other:
# model.fit(X_train, y_train)
# #reporting the classification accuracy on the testing set
# print "Train accuracy %f" %model.score(X_train,y_train)
# print "Test accuracy %f"%model.score(X_test,y_test)
