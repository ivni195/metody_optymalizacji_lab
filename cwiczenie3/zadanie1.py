from scipy.linalg import circulant
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cvxpy as cvx

# dane wejściowe

source_path = 'Data01.mat'
mat = scipy.io.loadmat(source_path)

t = mat['t']
y = mat['y']
N = y.shape[0]

# minimalizacja z parametrem q

y_pred = cvx.Variable((N,1))
q = 1.55
c1 = cvx.norm1(y_pred[1:] - y_pred[0:-1]) <= q

diff = cvx.Minimize(cvx.power(cvx.norm2(y_pred - y), 2))
problem = cvx.Problem(diff, [c1])
problem.solve(solver="ECOS")
plt.plot(t, y_pred.value, c='r', label='Predicted $y$')
plt.title("Optimization with constrait of estimated q")
plt.xlabel("$t$")
plt.ylabel("$y$")
plt.scatter(t, y, label='Original $y$ data points')
plt.legend()
plt.show()

# minimalizacja z parametrem r

y_pred = cvx.Variable((N,1))
r = 1.55

diff = cvx.Minimize(cvx.power(cvx.norm2(y_pred - y), 2) + cvx.norm1(y_pred[1:] - y_pred[0:-1]) * r)
problem = cvx.Problem(diff, [])
problem.solve(solver="ECOS")
plt.plot(t, y_pred.value, c='r', label='Predicted $y$')
plt.title("Optimization with estimated r")
plt.xlabel("$t$")
plt.ylabel("$y$")
plt.scatter(t, y, label='Original $y$ data points')
plt.legend()
plt.show()