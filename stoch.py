import numpy as np
import random
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats
import math
import pdb


# make y pred function
def predict(row, w):
    yhat = w[0]
    m = 6
    for i in range(len(row)-1):
        yhat += w[i + 1] * row[i]

    return yhat  #returns yhat (estimated y)

# stochastic gradient descent func uses (x, alpha, iterations)
def sgd(x, alpha, iterations, ep=0.0000000001):
    weight = [0.0 for i in range(len(x[0]))] #set weight
    for iter in range(iterations):
        y = np.random.randint(20, size=6) #set random y np array
        J = 0
        for row in x:
            yhat = predict(row, weight)
            err = yhat - row[-1]  # estimate error
            J = err**2
            c = abs(J-err)
            weight[0] = weight[0] - alpha * err
            for i in range(len(row)-1):
                weight[i + 1] = weight[i + 1] - alpha * err * row[i]
        print('Iter: %d | yHat = %.3f | Error = %.5f | Differ = %.10f ' %(iter, yhat, J, c))

    return weight



if __name__ == '__main__':
    x = np.array([[2,5], [4,7], [6,14], [7,14], [8,17], [10, 19]]) #declare x array
    alpha = 0.0001  #learning rate
    ep = 0.0000000001 #convergence
    iterations = 200000 #iter count
    w = [0.25, 0.25] #inital weights

    new_weights = sgd(x, alpha, iterations, ep)
    print('New Weights: w0 = %.5f w1 = %.5f' %(new_weights[0], new_weights[1]))  #new weights

    x, y = make_regression(n_samples=6, n_features=1, n_informative=1, random_state=0, noise=35)
    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x[:,0], y) #use stats
    print('Intercept = %s slope = %s' %(intercept, slope))

    x5 = intercept * 5 + slope
    print('When x = 5, y = %s' %(x5))
    neg100 = intercept * -100 + slope
    print('When x = -100, y = %s' %(neg100))
    x100 = intercept * 100 + slope
    print('When x = 100, y = %s' %(x100))

    #plot
    for i in range(x.shape[0]):
        y_pred = new_weights[0] + new_weights[1]*x

    pylab.plot(x,y,'o')
    pylab.plot(x,y_pred,'k-')
    pylab.show()
