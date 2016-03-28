""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_digits()
print data.DESCR
num_trials = 100
train_percentages = range(5,95,5)
test_accuracies = numpy.zeros(len(train_percentages))

# train a model with training percentages between 5 and 90 (see train_percentages) and evaluate
# the resultant accuracy.
# You should repeat each training percentage num_trials times to smooth out variability
# for consistency with the previous example use model = LogisticRegression(C=10**-10) for your learner
#result_data_train = []
result_data_test = []
for percentage in train_percentages:
    n = 0
    #train_data = []
    test_data = []
    while n <= num_trials :
        n = n + 1
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size= percentage)
        model = LogisticRegression(C=10**-10)
        model.fit(X_train, y_train)
        #model_score_train =model.score(X_train,y_train)
        model_score_test = model.score(X_test,y_test)
       # train_data.append(model_score_train)
        test_data.append(model_score_test) # this is a list of floats
    #train_ave= numpy.mean(train_data)
    test_ave = numpy.mean(test_data)
    #result_data_train.append(train_ave)
    result_data_test.append(test_ave)
    test_accuracies = result_data_test



fig = plt.figure()
plt.plot(train_percentages, test_accuracies, label = 'testing data')
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.legend()
plt.show()
