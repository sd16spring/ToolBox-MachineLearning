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
print train_percentages

digits = load_digits()
print digits.DESCR
fig = plt.figure()
for i in range(10):
    subplot = fig.add_subplot(5,2,i+1)
    subplot.matshow(numpy.reshape(digits.data[i],(8,8)),cmap='gray')

plt.show()

# train a model with training percentages between 5 and 90 (see train_percentages) and evaluate
# the resultant accuracy.
# You should repeat each training percentage num_trials times to smooth out variability
# for consistency with the previous example use model = LogisticRegression(C=10**-10) for your learner
test_accuracies = []
n = 0
while n<len(train_percentages):
	test_accuracy_total = 0
	for i in range(num_trials):
		data = load_digits()
		X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size = train_percentages[n]/100.0)
		model = LogisticRegression(C=10**-1)
		model.fit(X_train, y_train)
		test_accuracy_total += model.score(X_test,y_test)
	test_accuracy = test_accuracy_total/num_trials
	test_accuracies.append(test_accuracy)	
	print "Train Size %f" %train_percentages[n] 
	print "Test accuracy %f" %test_accuracy
	n += 1


fig = plt.figure()
plt.plot(train_percentages, test_accuracies)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.title('Accuracy on Test Set as a Function of Percentage of Data Used for Training')
plt.show()
