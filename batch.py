import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats
import math

# x = np.array([2,4,6,7,8,10])
# y = np.array([5,7,14,14,17,19])

# inital weights:
#       w0 = 0.25
#       w1 = 0.25

def batch_gradient_descent(alpha, x, y, ep=0.0001, iterations=10000):
    converged = False
    iter = 0
    m = x.shape[0]  # num of samples

    #initial weights
    w = np.array([0.25, 0.35])

    wx = sum([(w[0] + w[1]*x[i] - y[i])**2 for i in range(m)])
    print(wx)
    hwx = [1/(1+exp(-wx)) for i in range(m)]


    J = sum([(w[0] + w[1]*x[i] - y[i])**2 for i in range(m)])

    # iterate loop
    while not converged:
        grad0 = 1.0/m * sum([(w[0] + w[1]*x[i] - y[i]) for i in range(m)])
        grad1 = 1.0/m * sum([(w[0] + w[1]*x[i] - y[i])*x[i] for i in range(m)])


        #update theta temp
        temp0 = w[0] - alpha * grad0
        temp1 = w[1] - alpha * grad1

        # update theta
        w0 = temp0
        w1 = temp1

        e = sum( [ (w0 + w1*x[i] - y[i])**2 for i in range(m)] )

        if abs(J-e) <= ep:
            print('Converged, iterations: ', iter, '!!')
            converged = True

            J = e #update error
            iter += 1 #update iter

            if iter == iterations:
                print('Max iterations')
                converged = True

    return w0,w1

if __name__ == '__main__':
    x = np.array([[2,5],
                 [4,7],
                 [6,14],
                 [7,14],
                 [8,17],
                 [10,19]], np.int32)

    x, y = make_regression(n_samples=100, n_features=1, n_informative=1, random_state=0, noise=35)

    print('x.shape = %s y.shape = %s' %(x.shape, y.shape))

    alpha = 0.0001 # learning rate
    ep = 0.01 # convergence criteria

    theta0, theta1 = batch_gradient_descent(alpha, x, y, ep, iterations=1000)
    print('theta0 = %s theta1 = %s' %(theta0, theta1))

    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x[:,0], y)
    print('intercept = %s slope = %s' %(intercept, slope))

    for i in range(x.shape[0]):
        y_predict = theta0 + theta1*x

    pylab.plot(x,y, 'o')
    pylab.plot(x,y_predict, 'k-')
    pylab.show()
    print('Done!')
