""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_digits()
num_trials = 10
train_percentages = map(lambda x: x/100.0, range(5,95,5))



def train(percent, trials):
    scores = []
    model = LogisticRegression(C=10**-10)
    for i in range(trials):
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=(percent)) #Split the data into test and training
        model.fit(X_train, y_train) #Train the model on the test set
        scores.append(model.score(X_test,y_test)) #store the results 
    return sum(scores)/float(trials) #return the average result

results = [train(percent, num_trials) for percent in train_percentages] #list comprehension to iterate through percentages
fig = plt.figure()
plt.plot(train_percentages, results)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.show()
