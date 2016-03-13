""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_digits()
# print data.DESCR
num_trials = 200
train_percentages = range(5,95,5)
test_accuracies = numpy.zeros(len(train_percentages))
standards = numpy.zeros(len(train_percentages))

test_data = [] #test_data will contain each test point.
               #The standard deviations will be graphed to see variability.
for i in range(len(train_percentages)):
    test_data.append(numpy.zeros(num_trials))

# train a model with training percentages between 5 and 90 (see train_percentages) and evaluate
# the resultant accuracy.
# You should repeat each training percentage num_trials times to smooth out variability
# for consistency with the previous example use model = LogisticRegression(C=10**-10) for your learner

# TODO: your code here

#first do a train, test, split for each percentage num_trials number of times.
#add data to test_accuracies, standards
#we will add scatter data later
for i in range(len(train_percentages)):
    percent = train_percentages[i] / 100.0

    for j in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                            data.target,
                                                            train_size=percent)
        model = LogisticRegression(C=10**-10)
        model.fit(X_train, y_train)
        test_data[i][j] = model.score(X_test,y_test) #add datapoint to test_data

    test_accuracies[i] = numpy.average(test_data[i])
    standards[i] = numpy.std(test_data[i])
    print i

#create data for scatter plot
scatterX = []
for i in train_percentages:
    for j in range(num_trials):
        scatterX.append(i)

scatterY = []
for i in test_data:
    for j in i:
        scatterY.append(j)
#end create data for scatter plot

fig = plt.figure("Test Accuracy")
plt.plot(train_percentages, test_accuracies)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')

fig2 = plt.figure("Standard Deviation")
plt.plot(train_percentages, standards)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Standard Deviation of Test Set')

fig3 = plt.figure("Scatter")
plt.scatter(scatterX, scatterY, s=.1)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.show()
