# Mini Project - MachineLearning

Before you start, make sure you have scikit-learn and all its dependencies:

    $ sudo apt-get install build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev libatlas3gf-base
    $ sudo apt-get install python-matplotlib
    $ sudo pip install -U scikit-learn
  
learning_curve.py is a simple script that takes in grayscale, 8x8 pixel numbers from the built-in scikit-learn database and classifies them with Multinomial Logistic Regression (MLR). 

MLR works by partitioning the digits into two sets: a training set and a testing set. The training set is used for inferring the relationship between the current 8x8 picture and the estimated number. The testing set evaluates the training set's performance and applies it to the untrained, testing set. Having two sets is essential for avoiding possible overfitting to the training set. 

The learning_curve() function then takes in the MLR results and plots the resulting curve.

The cross_validation() function allows us to test a variety of C values to see which one works best. In our particular example, we tested 5 values: 10<sup>-4</sup>, 10<sup>-2</sup>, 1, 10<sup>2</sup>, and 10<sup>4</sup>.

Initial code from https://sites.google.com/site/sd16spring/home/project-toolbox/machine-learning
