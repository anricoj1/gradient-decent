import numpy as np
import random
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats
import math

def stochastic(alpha, x, y, ep, max_iter=10000000):
    converged = False
    m = x.shape[0] # number of samples
    iter = 0

    x_transpose = x.transpose()

    w0 = 0.25
    w1 = 0.3

    while not converged:
        wx = get_wx(x, y, m, w0, w1)
        hwx = get_hwx(wx, m)

    return get_grad(x, y, w0, w1, m, hwx, iter)

def get_wx(x, y, m, w0, w1):
    m = x.shape[0]
    for i in range(m):
        wx = (w0 + w1*x[i] - y)**2

    return wx


def get_hwx(wx, m):
    m = x.shape[0]
    for i in range(m):
        hwx = 1/(1 + math.exp(-wx[i]))

    return hwx

def get_grad(x, y, w0, w1, m, hwx, iter):
    m = x.shape[0]
    cost = sum([(w0 + w1*x[i] - y[i]) for i in range(m)])
    for i in range(m):
        g0 = w0 + 0.0001 * (y - hwx[i])
        g1 = w1 + 0.0001 * (y - hwx[i]) * x[i]

        tmp0 = w0 + alpha * g0
        tmp1 = w1 + alpha * g1

        w0 = tmp0
        w1 = tmp1

        err = sum( [ (w0 + w1*x[i] - y[i])**2 for i in range(m)] )

        c = abs(cost - err)

        if c <= 0.0000000001:
            print('Converged after %s Iterations! With Convergence = %.10f' %(iter, c))
            converged = True

        J = err
        iter += 1

        if iter == 100000000:
            print('Max Iter')
            converged = True


    return w0, w1

if __name__ == '__main__':

    x = np.array([2,4,6,7,8,10])

    x, y = make_regression(n_samples=100, n_features=1, n_informative=1,
                        random_state=0, noise=35)

    print('x.shape = %s y.shape = %s' %(x.shape, y.shape))

    alpha = 0.0001
    ep = 0.0000000001

    theta0, theta1 = stochastic(alpha, x, y, ep, max_iter=1000000)
    print('w0 = %.10f w1 = %.10f' %(theta0, theta1))

    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x[:,0], y)

    print('Intercept = %s slope = %s' %(intercept, slope))
    x5 = intercept * 5 + slope
    print('When x = 5, y = %s' %(x5))
    neg100 = intercept * -100 + slope
    print('When x = -100, y = %s' %(neg100))
    x100 = intercept * 100 + slope
    print('When x = 100, y = %s' %(x100))

    for i in range(x.shape[0]):
        y_pred = theta0 + theta1*x

    pylab.plot(x,y,'o')
    pylab.plot(x,y_pred,'k-')
    pylab.show()
    print('Done!')
