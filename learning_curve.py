""" Exploring learning curves for classification of handwritten digits
 
- train a model with training percentages between 5 and 90 (see train_percentages)
- evaluate the resultant accuracy
- repeat each training percentage num_trials times to smooth out variabilityfor consistency with the previous example use model = LogisticRegression(C=10**-10) for your learner

 """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import numpy

digits = load_digits()
print digits.DESCR
fig = plt.figure()
for i in range(10):
    subplot = fig.add_subplot(5,2,i+1)
    subplot.matshow(numpy.reshape(digits.data[i],(8,8)),cmap='gray')

plt.show()

data = load_digits()
print data.DESCR
num_trials = 10
train_percentages = range(5,95,5)
#test_accuracies = numpy.zeros(len(train_percentages))
test_accuracies = []
test_accuracies_avg = []
test_accuracies_plot = []


# loop through each percentage for the train size
for percent in train_percentages:
	# repeat each train size 'num_trial' times
	for trial in range(num_trials):

		X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size= percent/100.0) # train size takes value between 0 and 1 (i.e. 0.5 = 50%)
		model = LogisticRegression(C=10**-10)
		model.fit(X_train, y_train)
		test_accuracies.append(model.score(X_test,y_test)) # add newest accuracy to list

	test_accuracies_avg = sum(test_accuracies)/num_trials # get average accuracy over all trials
	test_accuracies_plot.append(test_accuracies_avg) # append average accuracy to list for plotting

print "Train accuracy %f" %model.score(X_train,y_train)
print "Test accuracy %f"%model.score(X_test,y_test)
print "Average test accuracy: ", test_accuracies_plot

fig = plt.figure()
plt.plot(train_percentages, test_accuracies_plot)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.show()