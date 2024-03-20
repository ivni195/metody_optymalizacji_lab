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

y_pred = cvx.Variable((N,1))
q = 1.55

# tworzenie i definiowanie macierzy A, c, b, x

first_row = np.zeros(N)
first_row[0] = -1
first_row[1] = 1
D = circulant(first_row).transpose()[:-1]

A = np.block([
    [np.identity(N), -np.identity(N), np.zeros((N, (N-1)))],
     [-np.identity(N), -np.identity(N), np.zeros((N, (N-1)))],
     [np.zeros((1, N)), np.zeros((1, N)), np.ones((N-1))],
     [-D, np.zeros(((N-1), N)), -np.identity((N-1))],
     [D, np.zeros(((N-1), N)), -np.identity((N-1))]
])

c = np.block([
    [np.zeros((N, 1))],
    [np.ones((N, 1))],
    [np.zeros((N-1, 1))]
])

b = np.block([
    [np.array(y).reshape(-1, 1)],
    [-np.array(y).reshape(-1, 1)],
    [np.array(q).reshape(-1, 1)],
    [np.zeros((N - 1, 1))],
    [np.zeros((N - 1, 1))]
])

xi = cvx.Variable((N, 1))
delta = cvx.Variable((N-1, 1))
x = cvx.vstack((y_pred, xi, delta))

# minimalizacja z parametrem q w postaci LP

diff = cvx.Minimize(c.transpose()@x)
problem = cvx.Problem(diff, [A@x <= b])
problem.solve(solver="ECOS")

plt.plot(t, y_pred.value, c='r', label='Predicted $y$')
plt.title("Optimization by LP with estimated q")
plt.xlabel("$t$")
plt.ylabel("$y$")
plt.scatter(t, y, label='Original $y$ data points')
plt.legend()
plt.show()