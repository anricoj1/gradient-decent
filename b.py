import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats
import math


def batch_gradient_descent(x,y):
    m = b = 0
    iterations = 1000
    n = len(x)
    alpha = 0.0001

    for i in range(iterations):
        y_pred = m * x + b
        cost = (1/n) * sum([value**2 for value in (y-y_pred)])
        md = -(2/n)*sum(x*(y-y_pred))
        bd = -(2/n)*sum(y-y_pred)
        m = m - alpha * md
        b = b - alpha * bd
        print("m {}, b {}, cost {}, iteration {}".format(m,b,cost,i))



x = np.array([2,4,6,7,8,10])
y = np.array([5,7,14,14,17,19])



batch_gradient_descent(x,y)
