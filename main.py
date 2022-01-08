import time

from matplotlib import pyplot as plt
import numpy as np
import math


def f(x):
    return x ** 3 - 2 * x - 10 + x ** 2


def derivative_f(x):
    return 3 * (x ** 2) - 2 + 2 * x


def rprop():
    x = 0.0
    y = 0.0

    learning_rate = 0.001
    gradient = 0

    delta = 5

    a = 1.2
    b = 0.5

    eta_max = 50
    eta_min = 0.000001

    grad_log = [0, 0, 0]
    X = []
    Y = []

    for i in range(10000):
        print('step {:d}: x = {:6f}, f(x) = {:6f}, gradient={:6f}'.format(i + 1, x, y, derivative_f(x)))

        if grad_log[-1] == grad_log[-3] and grad_log[-1] != 0:
            print("Сходится за " + str(i + 1) + " шагов к точке x = " + str(x))
            break
        else:
            if gradient * derivative_f(x) > 0:
                delta = min(delta * a, eta_max)
            elif gradient * derivative_f(x) < 0:
                delta = max(delta * b, eta_min)

            if derivative_f(x) < 0:
                x += delta * learning_rate
            elif derivative_f(x) > 0:
                x -= delta * learning_rate

            gradient = derivative_f(x)
            grad_log.append(gradient)
            y = f(x)
            X.append(x)
            Y.append(y)

    return X, Y


def desend():
    x = 0
    learning_rate = 0.01
    gradient = 0
    X = []
    Y = []

    for i in range(10000):
        print('step {:d}: x = {:6f}, f(x) = {:6f}, gradient={:6f}'.format(i + 1, x, f(x), gradient))

        if 0.00001 < abs(gradient) < 0.0001:
            print("Сходится за " + str(i + 1) + " шагов к точке x = " + str(x))
            break
        else:
            gradient = derivative_f(x)
            x = x - learning_rate * gradient
            y = f(x)
            X.append(x)
            Y.append(y)

    return X, Y


def fast_desend():
    x = 0

    gradient = 0
    X = []
    Y = []
    mn = 100

    for i in range(100000):
        print('step {:d}: x = {:6f}, f(x) = {:6f}, gradient={:6f}'.format(i + 1, x, f(x), gradient))

        if 0.00001 < abs(gradient) < 0.0001:
            print("Сходится за " + str(i + 1) + " шагов к точке x = " + str(x))
            break
        else:
            gradient = derivative_f(x)
            learning_rate = 1 / min(i + 1, mn)
            x = x - learning_rate * np.sign(gradient)
            y = f(x)
            X.append(x)
            Y.append(y)

    return X, Y


def rmsprop():
    x = 0.0
    y = 0.0

    learning_rate = 0.01
    gradient = 0

    beta = 0.9

    Egk = 0

    X = []
    Y = []

    for i in range(100000):
        print('step {:d}: x = {:6f}, f(x) = {:6f}, gradient={:6f}'.format(i + 1, x, y, gradient))
        if 0.00001 < abs(gradient) < 0.0001:
            print("Сходится за " + str(i + 1) + " шагов к точке x = " + str(x))
            break
        else:
            gradient = derivative_f(x)
            Egk = beta * Egk + (1 - beta) * (gradient ** 2)
            x = x - learning_rate * gradient / math.sqrt(Egk)
            y = f(x)
            X.append(x)
            Y.append(y)
    return X, Y


if __name__ == '__main__':
    start = time.time() * 1000000000
    X, Y = rmsprop()
    #X, Y = rprop()
    #X, Y = desend()
    #X, Y = fast_desend()
    end = time.time() * 1000000000
    print("Время работы - " + str(end - start))
    plt.plot(X[:-1], Y[:-1])
    plt.scatter(X[-1], Y[-1], color= "red", alpha =1)
    plt.show()
