""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_digits()
print data.DESCR

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=0.5)
model = LogisticRegression(C=10**-10)
model.fit(X_train, y_train)
print "Train accuracy %f" %model.score(X_train,y_train)
print "Test accuracy %f"%model.score(X_test,y_test)

# fig = plt.figure()
# for i in range(10):
#     subplot = fig.add_subplot(5,2,i+1)
#     subplot.matshow(np.reshape(data.data[i],(8,8)), cmap='gray')
# plt.show()

num_trials = 50 #200 is super accurate
index = 0

train_percentages = range(5,95,5)
test_accuracies = []

X_train_total = {}
X_test_total = {}
y_train_total = {}
y_test_total = {}


#loop through different c values, if you want.
for c in [0]:#ange(0, -20, -5):
    fig = plt.figure()
    Cval = 10**c
    for i in range(num_trials):
        for p in train_percentages:
            results = train_test_split(data.data, data.target, train_size=p/100.0)
            if i == 0:
            
                X_train_total[p] = results[0]
                X_test_total[p] = results[1]
                y_train_total[p] = results[2]
                y_test_total[p] = results[3]
            else:
                X_train_total[p] = np.add(X_train_total.get(p, 0), results[0])
                X_test_total[p] = np.add(X_test_total.get(p, 0), results[1])
                y_train_total[p] = np.add(y_train_total.get(p, 0), results[2])
                y_test_total[p] = np.add(y_test_total.get(p, 0), results[3])

            
            model = LogisticRegression(C=Cval)
            model.fit(results[0], results[2])
            test_accuracies.append(model.score(results[1],results[3]))

        plt.plot(train_percentages, test_accuracies)
        test_accuracies=[]

    # print '***** {}% *****'.format(p)
    # print "Train accuracy %f" %model.score(X_train,y_train)
    # print "Test accuracy %f"%model.score(X_test,y_test)

#plt.plot(train_percentages, test_accuracies)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')

fig2 = plt.figure()
test_accuracies_average = []
print len(X_train_total)
print X_train_total[5]
for p in train_percentages:
    X_train = X_train_total[p]/num_trials
    X_test = X_test_total[p]/num_trials
    y_train = y_train_total[p]/num_trials
    y_test = y_test_total[p]/num_trials

    model = LogisticRegression(C=1.0)
    model.fit(X_train, y_train)
    test_accuracies_average.append(model.score(X_test,y_test))
plt.plot(train_percentages, test_accuracies_average)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.show()

