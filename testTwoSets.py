from sklearn.datasets import *
from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegression

data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=0.5)
model = LogisticRegression(C=10**-10)
model.fit(X_train, y_train)
print "Train accuracy %f" %model.score(X_train,y_train)
print "Test accuracy %f"%model.score(X_test,y_test)
