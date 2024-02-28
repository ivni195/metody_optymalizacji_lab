import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data01.csv", header=None)
x = np.array(df[0]).reshape(-1, 1)
y = np.array(df[1]).reshape(-1, 1)


plt.scatter(x, y)
plt.xlabel("$x_i, i = 1,..., N$")
plt.ylabel("$y_i, i = 1,..., N$")
plt.grid()
plt.show()


# Create a matrix [[x0, 1], [x1, 1], [x2, 1], ... ] 
A = np.hstack((x, x ** 0))

# Create vectors theta = [a, b] that will be optimized
theta_ls = cp.Variable([2, 1])
theta_lp = cp.Variable([2, 1])

# Optimize with LS
obj = cp.Minimize(cp.norm(A @ theta_ls - y, 2))
cp.Problem(obj,[]).solve(solver=cp.ECOS)

# Optimize with LP 
obj = cp.Minimize(cp.norm(A @ theta_lp - y, 1))
cp.Problem(obj,[]).solve(solver=cp.ECOS)

print(f'Metoda LS: a = {theta_ls.value[0][0]:4f}, b = {theta_ls.value[1][0]:4f}')
print(f'Metoda LP: a = {theta_lp.value[0][0]:4f}, b = {theta_lp.value[1][0]:4f}')

# Create plots from opimized a and b values
x_reg = np.linspace(-2 ,12, 1000)
y_reg_ls = x_reg * theta_ls.value[0][0] + theta_ls.value[1][0]
y_reg_lp = x_reg * theta_lp.value[0][0] + theta_lp.value[1][0]

plt.scatter(x, y)
plt.plot(x_reg, y_reg_ls, c='r')
plt.plot(x_reg, y_reg_lp, c='k')
plt.legend(["$(x_i, y_i), i = 1,...,N$","$y = ax + b$, (LS)", "$y = ax + b$, (LP)"])
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.xlim(-2, 12)
plt.ylim(0, 12)
plt.grid()
plt.show()

