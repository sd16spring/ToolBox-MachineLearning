""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

""" recognize images of handwritten digits """
# digits = load_digits()
# print digits.DESCR
# fig = plt.figure()
# for i in range(10):
#     subplot = fig.add_subplot(5,2,i+1)
#     subplot.matshow(numpy.reshape(digits.data[i],(8,8)),cmap='gray')

# plt.show()

""" initial conditions """
data = load_digits()
# print data.DESCR
num_trials = 100
train_percentages = range(5,95,5)
test_accuracies = numpy.zeros(len(train_percentages))

# train a model with training percentages between 5 and 90 (see train_percentages) and evaluate
# the resultant accuracy.
# You should repeat each training percentage num_trials times to smooth out variability
# for consistency with the previous example use model = LogisticRegression(C=10**-10) for your learner

""" partition data into two sets--training set and testing set 
	vary training set size vs. testing set size and plot resulting curve """
for i in range(len(train_percentages)):
    trial_accuracies= []
    for j in range(num_trials):
        x_train,x_test,y_train,y_test = train_test_split(data.data, data.target, train_size=train_percentages[i]/200.0)                                                      
        model = LogisticRegression(C=10 ** -10)
        model.fit(x_train, y_train)
        accur_score = model.score(x_test,y_test)
        trial_accuracies.append(accur_score)

    test_accuracies[i] = sum(trial_accuracies) / num_trials

fig = plt.figure()
plt.plot(train_percentages, test_accuracies)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.show()

