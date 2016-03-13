from sklearn.datasets import *
import matplotlib.pyplot as plt
import numpy

digits = load_digits()
print digits.DESCR
fig = plt.figure()
for i in range(10):
    subplot = fig.add_subplot(5,2,i+1)
    subplot.matshow(numpy.reshape(digits.data[i],(8,8)),cmap='gray')

plt.show()
