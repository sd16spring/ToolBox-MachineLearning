import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_digits()
print data.DESCR
num_trials = 50
train_percentages = range(1,99,1)
test_accuracies = numpy.zeros(len(train_percentages))

# train a model with training percentages between 5 and 90 (see train_percentages) and evaluate
# the resultant accuracy.
# You should repeat each training percentage num_trials times to smooth out variability
# for consistency with the previous example use model = LogisticRegression(C=10**-10) for your learner

for i, percent in enumerate(train_percentages):
	for j in range(num_trials):
		X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=percent/100.0)
		model = LogisticRegression(C=10**-10)
		model.fit(X_train, y_train)
		print "Train accuracy %f" %model.score(X_train,y_train)
		print "Test accuracy %f"%model.score(X_test,y_test)
		test_accuracies[i] += model.score(X_test,y_test)
	test_accuracies[i] /= num_trials

fig = plt.figure()
plt.plot(train_percentages, test_accuracies*100)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.axis([0,100,0,100])
plt.show()