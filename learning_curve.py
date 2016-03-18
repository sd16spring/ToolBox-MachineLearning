""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_digits()
# print data.DESCR
num_trials = 25
train_percentages = range(5,90,5)

test_accuracies = []

for num in train_percentages:
	trainaccuracy = []
	testaccuracy = []

	for num1 in range(0,num_trials):

		x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, train_size=(float(num)/100.0))
		model = LogisticRegression(C=10**-10)
		model.fit(x_train, y_train)
		trainaccuracy.append(model.score(x_train,y_train))
		testaccuracy.append(model.score(x_test,y_test))
	print num
	trainaccuracyavg = np.mean(trainaccuracy)
	testaccuracyavg = np.mean(testaccuracy)
	test_accuracies.append(np.mean(testaccuracy))

fig = plt.figure()
plt.plot(train_percentages, test_accuracies)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.show()
