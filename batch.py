import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats
import math


def batch_gradient_descent(alpha, x, y, ep=0.0000000001, max=1000000):
    converged = False # for while loop
    iter = 0 # starting iteration
    m = x.shape[0] # number of samples for range


    #inital weights
    w = [0.25, 0.25]

    # total error
    J = sum([(w[0] + w[1]*x[i] - y[i])**2 for i in range(m)])

    # Iterate until we see convergence or "ep"
    while not converged:
        # for each sample we find the gradient 1/m * w0 + w1(xi) - y(i)
        g0 = 1.0/m * sum([(w[0] + w[1]*x[i] - y[i]) for i in range(m)])
        g1 = 1.0/m * sum([(w[0] + w[1]*x[i] - y[i])*x[i] for i in range(m)])

        # update weights
        tmp0 = w[0] - alpha * g0
        tmp1 = w[1] - alpha * g1

        w[0] = tmp0
        w[1] = tmp1

        # mean squared error
        err = sum([(y[i] - (w[1]*x[i] + w[0]))**2 for i in range(m)])

        c = abs(J-err)

        print('Iteration = %s | error = %s | EP = %.10f' %(iter, err, c))

        if c <= ep:
            print('Converged after %s Iterations! With Convergence = %.10f' %(iter, c))
            converged = True

        J = err #update error
        iter += 1 #increase iteration

        # if we reach max iterations of 1,000,000 converged is True
        if iter == max:
            print('Max Iterations Reached!')
            converged = True


    return w[0], w[1]

if __name__ == '__main__':

    alpha = 0.0001 #learning rate
    ep = 0.0000000001
    x = np.array([2,4,6,7,8,10])
    y = np.array([5,7,14,14,17,19])

    w0, w1 = batch_gradient_descent(alpha, x, y, ep, max=1000000) #call batch to determine w0,w1
    print('w0 = %.10f w1 = %.10f' %(w0, w1))

    x, y = make_regression(n_samples=6, n_features=1, n_informative=1, random_state=0, noise=35)

    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x[:,0], y)
    print('Intercept = %s slope = %s' %(intercept, slope))

    x5 = intercept * 5 + slope
    print('When x = 5, y = %s' %(x5))
    neg100 = intercept * -100 + slope
    print('When x = -100, y = %s' %(neg100))
    x100 = intercept * 100 + slope
    print('When x = 100, y = %s' %(x100))

    for i in range(x.shape[0]):
        y_pred = w0 + w1*x

    pylab.plot(x,y,'o')
    pylab.plot(x,y_pred,'k-')
    pylab.show()
