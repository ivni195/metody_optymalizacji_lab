import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from mpl_toolkits.mplot3d import Axes3D

P = 1 / 8 * np.array([[7, np.sqrt(3)], [np.sqrt(3), 5]])

# Zadanie 2 - funkcja
X1 = np.linspace(-0, 2, 100)
X2 = np.linspace(-0.5, 2, 100)
x1, x2 = np.meshgrid(X1, X2)
t = 1
f = t * (np.exp(x1 + 3 * x2 - 0.1) + np.exp(-x1 - 0.1)) - np.log(1 - (1/8 * ((-1 + x1) * (7 * (-1 + x1) + np.sqrt(3) * (-1 + x2)) + (np.sqrt(3) * (-1 + x1) + 5 * (-1 + x2)) * (-1 + x2))))
a = plt.contour(x1, x2, f, [2.5, 3.62, 5.34, 7.59, 19, 50, 200, 600], colors='k')
# plt.scatter(x_path[:, 0], x_path[:, 1])
plt.clabel(a, fontsize=9, inline=True)
plt.title("Zadanie 2 - Funkcja")
plt.xlabel("$x1$")
plt.ylabel("$x2$")
plt.show()


plt.contour(x1, x2, 1 - (1/8 * ((-1 + x1) * (7 * (-1 + x1) + np.sqrt(3) * (-1 + x2)) + (np.sqrt(3) * (-1 + x1) + 5 * (-1 + x2)) * (-1 + x2))), [-1, 0, 0.5], colors='k')

plt.show()

# Zadanie 2 - zmienne

def my_log(a_):
    return np.log(a_) if a_.all() > 0 else np.inf * np.ones(a_.shape)

def f_func(x1_, x2_):
    return t * (np.exp(x1_ + 3 * x2_ - 0.1) + np.exp(-x1_ - 0.1)) - my_log(1 - (1/8 * ((-1 + x1_) * (7 * (-1 + x1_) + np.sqrt(3) * (-1 + x2_)) + (np.sqrt(3) * (-1 + x1_) + 5 * (-1 + x2_)) * (-1 + x2_))))


def g(x1_, x2_):
    x_vec = np.array([x1_, x2_])
    xc = np.array([1, 1])
    a = t * np.array([
        -np.exp(-0.1 - x1_) + np.exp(-0.1 + x1_ + 3 * x2_),
         3 * np.exp(-0.1 + x1_ + 3 * x2_)
    ]) + (2 * P @ (x_vec - xc)) / (1 - (x_vec - xc).transpose() @ P @ (x_vec - xc))
    return a if a.any() != np.nan else 10**10 * np.ones(2)
    

def hess(x1_, x2_):
    x_vec = np.array([x1_, x2_])
    xc = np.array([1, 1])
    return t * np.array([
        [np.exp(-0.1 - x1_) + np.exp(-0.1 + x1_ + 3 * x2_), 3 * np.exp(-0.1 + x1_ + 3 * x2_)],
        [3 * np.exp(-0.1 + x1_ + 3 * x2_), 9 * np.exp(-0.1 + x1_ + 3 * x2_)]
    ]) + (2 * P) / (1 - (x_vec - xc).transpose() @ P @ (x_vec - xc)) + (4 * P @ (x_vec - xc) * (x_vec - xc).transpose() @ P) / ((1 - (x_vec - xc).transpose() @ P @ (x_vec - xc)) * (1 - (x_vec - xc).transpose() @ P @ (x_vec - xc)))

def v(x1_, x2_):
    return -np.linalg.inv(hess(x1_, x2_)) @ g(x1_, x2_)


# Zadanie 2 - Metoda Newtona z tłumieniem (t=0.1)

t = 0.1
x = np.array([1, 1])
x_path = []
delta = -g(*x).transpose() @ v(*x)
epsilon = 1e-4
alpha = 0.3
beta = 0.8
x_path = []

while delta > epsilon:
    x_path.append(x)
    s = 1
    while f_func(*(x + s * v(*x))) > f_func(*x) + s * alpha * g(*x).transpose() @ v(*x):
        s *= beta
    x = x + s * v(*x)
    delta = -g(*x).transpose() @ v(*x)
    
x_path = np.array(x_path)
print(f"Steps: {x_path.shape[0]}\nx_optimal = {x}")

X1 = np.linspace(-0, 2, 100)
X2 = np.linspace(-0.5, 2, 100)
x1, x2 = np.meshgrid(X1, X2)
f = t * (np.exp(x1 + 3 * x2 - 0.1) + np.exp(-x1 - 0.1)) - my_log(1 - (1/8 * ((-1 + x1) * (7 * (-1 + x1) + np.sqrt(3) * (-1 + x2)) + (np.sqrt(3) * (-1 + x1) + 5 * (-1 + x2)) * (-1 + x2))))
a = plt.contour(x1, x2, f, [1.5, 3.62, 5.34, 7.59, 19, 50, 200, 600], colors='k')
plt.scatter(x_path[:, 0], x_path[:, 1])
plt.clabel(a, fontsize=9, inline=True)
plt.title("Zadanie 2 - Metoda Newtona z tłumieniem ($t=0.1$)")
plt.xlabel("$x1$")
plt.ylabel("$x2$")
plt.show()

# Zadanie 2 - Metoda Newtona z tłumieniem (t=1)

t = 1
x = np.array([1, 1])
x_path = []
delta = -g(*x).transpose() @ v(*x)
epsilon = 1e-4
alpha = 0.3
beta = 0.8
x_path = []

while delta > epsilon:
    x_path.append(x)
    s = 1
    while f_func(*(x + s * v(*x))) > f_func(*x) + s * alpha * g(*x).transpose() @ v(*x):
        s *= beta
    x = x + s * v(*x)
    delta = -g(*x).transpose() @ v(*x)
    
x_path = np.array(x_path)
print(f"Steps: {x_path.shape[0]}\nx_optimal = {x}")

X1 = np.linspace(-0, 2, 100)
X2 = np.linspace(-0.5, 2, 100)
x1, x2 = np.meshgrid(X1, X2)

f = t * (np.exp(x1 + 3 * x2 - 0.1) + np.exp(-x1 - 0.1)) - my_log(1 - (1/8 * ((-1 + x1) * (7 * (-1 + x1) + np.sqrt(3) * (-1 + x2)) + (np.sqrt(3) * (-1 + x1) + 5 * (-1 + x2)) * (-1 + x2))))
a = plt.contour(x1, x2, f, [1.5, 3.62, 5.34, 7.59, 19, 50, 200, 600], colors='k')
plt.scatter(x_path[:, 0], x_path[:, 1])
plt.title("Zadanie 2 - Metoda Newtona z tłumieniem ($t=1$)")
plt.clabel(a, fontsize=9, inline=True)
plt.xlabel("$x1$")
plt.ylabel("$x2$")
plt.show()

t = 10
x = np.array([1, 1])
x_path = []
delta = -g(*x).transpose() @ v(*x)
epsilon = 1e-4
alpha = 0.3
beta = 0.8
x_path = []

while delta > epsilon:
    
    x_path.append(x)
    s = 1
    while f_func(*(x + s * v(*x))) > f_func(*x) + s * alpha * g(*x).transpose() @ v(*x):
        s *= beta
    x = x + s * v(*x)
    delta = -g(*x).transpose() @ v(*x)
    
x_path = np.array(x_path)
print(f"Steps: {x_path.shape[0]}\nx_optimal = {x}")

X1 = np.linspace(-0, 2, 100)
X2 = np.linspace(-0.5, 2, 100)
x1, x2 = np.meshgrid(X1, X2)

f = t * (np.exp(x1 + 3 * x2 - 0.1) + np.exp(-x1 - 0.1)) - my_log(1 - (1/8 * ((-1 + x1) * (7 * (-1 + x1) + np.sqrt(3) * (-1 + x2)) + (np.sqrt(3) * (-1 + x1) + 5 * (-1 + x2)) * (-1 + x2))))
a = plt.contour(x1, x2, f, [1.5, 3.62, 5.34, 7.59, 19, 50, 200, 600], colors='k')
plt.scatter(x_path[:, 0], x_path[:, 1])
plt.clabel(a, fontsize=9, inline=True)
plt.xlim((-0, 2))
plt.ylim((-0.5, 2))
plt.title("Zadanie 2 - Metoda Newtona z tłumieniem ($t=10$)")
plt.xlabel("$x1$")
plt.ylabel("$x2$")
plt.show()
