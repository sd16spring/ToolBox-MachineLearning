import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_digits()
print data.DESCR
num_trials = 500
train_percentages = range(5,95,1)
test_accuracies = []
 
# train a model with training percentages between 5 and 90 (see train_percentages) and evaluate
# the resultant accuracy.
# You should repeat each training percentage num_trials times to smooth out variability
# for consistency with the previous example use model = LogisticRegression(C=10**-10) for your learner

for train_percentage in train_percentages: # For each percentage in train_percentages (goes from 5% to 95% in increments of 1%)
    running_average_variable = 0
    for i in range(0,num_trials): # Run the number of trials specified
       X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size = train_percentage)
       model = LogisticRegression(C=10**-10)
       model.fit(X_train, y_train)
       running_average_variable += model.score(X_test, y_test) # Add up all of the running averages
    test_accuracies.append(running_average_variable/num_trials) # divide running average by the number of trials to attain actual average

fig = plt.figure()
plt.plot(train_percentages, test_accuracies)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.show() 