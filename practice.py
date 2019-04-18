import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats


def batch_gradient_descent(alpha, x, y, numIter):
    m = x.shape[0] #sample size 6
    w = np.array([0.25,0.30])
    theta = np.ones(2)
    x_transpose = x.transpose()

    for iter in range(0, numIter):
        print(w[0:])
        h = np.dot(x, w[0:])
        loss = h - y
        J = np.sum(loss ** 2) / (2 * m)
        print('iter %s | J: %.3f' %(iter, J))
        batch = np.dot(x_transpose, loss) / m
        theta = theta - alpha * batch
    return theta

if __name__ == '__main__':

    x, y = make_regression(n_samples=6, n_features=1, n_informative=1, random_state=0, noise=35)

    m, n = np.shape(x)

    x = np.array([[2,5], [4,7], [6,14], [7,14], [8,17], [10,19]])


    alpha = 0.0001

    theta = batch_gradient_descent(alpha, x, y, 1)

    for i in range(x.shape[1]):
        y_predict = theta[0] + theta[1]*x
    pylab.plot(x[:,1],y,'o')
    pylab.plot(x,y_predict, 'k-')
    pylab.show()
    print('Done')
