""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_digits()
# print data.DESCR
num_trials = 20
train_percentages = range(5,95,5)
test_accuracies = numpy.zeros(len(train_percentages))

# train a model with training percentages between 5 and 90 (see train_percentages) and evaluate
# the resultant accuracy.
# You should repeat each training percentage num_trials times to smooth out variability
# for consistency with the previous example use model = LogisticRegression(C=10**-10) for your learner


def train_test(percent):
	'''partitions data into training and testing sets, using these groups to train and test the data,
	and returns tesing accuracy
	percent: percent of data partitioned for training'''
	X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size = percent/100.0)
	model = LogisticRegression(C=10**0)
	model.fit(X_train, y_train)
	return model.score(X_test,y_test)

for i in enumerate(train_percentages):
	t = 0
	for j in range(num_trials):
		t += train_test(i[1])/num_trials #averages accuracies for each percentage
	test_accuracies[i[0]] = t



fig = plt.figure()
# for i in range(10):
# 	subplot = fig.add_subplot(5,2,i+1)
# 	subplot.matshow(numpy.reshape(data.data[i], \
# 		(8,8)), cmap='gray')
plt.plot(train_percentages, test_accuracies)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.title(str(num_trials))
plt.show()
