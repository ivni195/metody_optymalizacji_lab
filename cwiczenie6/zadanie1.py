import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from mpl_toolkits.mplot3d import Axes3D

P = 1 / 8 * np.array([[7, np.sqrt(3)], [np.sqrt(3), 5]])

# Zadanie 1 - funkcja
X1 = np.linspace(-3, 2.5, 100)
X2 = np.linspace(-2, 2, 100)
x1, x2 = np.meshgrid(X1, X2)
f = np.exp(x1 + 3 * x2 - 0.1) + np.exp(-x1 - 0.1) + 1/8 * ((-1 + x1) * (7 * (-1 + x1) + np.sqrt(3) * (-1 + x2)) + (np.sqrt(3) * (-1 + x1) + 5 * (-1 + x2)) * (-1 + x2))
a = plt.contour(x1, x2, f, [2.5, 3.62, 5.34, 7.59, 19, 50, 200, 600], colors='k')
plt.title("Zadanie 1 - Funkcja")
plt.xlabel("$x1$")
plt.ylabel("$x2$")
plt.clabel(a, fontsize=9, inline=True)
plt.show()

# Zadanie 1 - zmienne

def f_func(x1_, x2_):
    return np.exp(x1_ + 3 * x2_ - 0.1) + np.exp(-x1_ - 0.1) + 1/8 * ((-1 + x1_) * (7 * (-1 + x1_) + np.sqrt(3) * (-1 + x2_)) + (np.sqrt(3) * (-1 + x1_) + 5 * (-1 + x2_)) * (-1 + x2_))

def g(x1_, x2_):
    return np.array([
        -np.exp(-0.1 - x1_) + np.exp(-0.1 + x1_ + 3 * x2_) + 1/8 * (14 * (-1 + x1_) + 2 * np.sqrt(3) * (-1 + x2_)),
         3 * np.exp(-0.1 + x1_ + 3 * x2_) + 1/8 * (2 * np.sqrt(3) * (-1 + x1_) + 10 * (-1 + x2_))
    ])

def hess(x1_, x2_):
    return np.array([
        [7/4 + np.exp(-0.1 - x1_) + np.exp(-0.1 + x1_ + 3 * x2_), np.sqrt(3)/4 + 3 * np.exp(-0.1 + x1_ + 3 * x2_)],
        [np.sqrt(3)/4 + 3 * np.exp(-0.1 + x1_ + 3 * x2_), 5/4 + 9 * np.exp(-0.1 + x1_ + 3 * x2_)]
    ])              

def v(x1_, x2_):
    return -np.linalg.inv(hess(x1_, x2_)) @ g(x1_, x2_)

# Zadanie 1 - Klasyczna metoda Newtona

x = np.array([2, -2])
x_path = []
delta = -g(*x).transpose() @ v(*x)
epsilon = 1e-4
while delta > epsilon:
    x_path.append(x)
    x = x + v(*x)
    delta = -g(*x).transpose() @ v(*x)

x_path = np.array(x_path)
print(f"Steps: {x_path.shape[0]}\nx_optimal = {x}")

# Zadanie 1 - Klasyczna metoda Newtona - wykres

X1 = np.linspace(-3, 2.5, 100)
X2 = np.linspace(-2, 2, 100)
x1, x2 = np.meshgrid(X1, X2)
f = np.exp(x1 + 3 * x2 - 0.1) + np.exp(-x1 - 0.1) + 1/8 * ((-1 + x1) * (7 * (-1 + x1) + np.sqrt(3) * (-1 + x2)) + (np.sqrt(3) * (-1 + x1) + 5 * (-1 + x2)) * (-1 + x2))
a = plt.contour(x1, x2, f, [2.5, 3.62, 5.34, 7.59, 19, 50, 200, 600], colors='k')
plt.scatter(x_path[:, 0], x_path[:, 1])
plt.title("Zadanie 1 - Klasyczna metoda Newtona")
plt.clabel(a, fontsize=9, inline=True)
plt.xlabel("$x1$")
plt.ylabel("$x2$")
plt.show()

# Zadanie 1 - Metoda Newtona z tłumieniem

x = np.array([2, -2])
x_path = []
delta = -g(*x).transpose() @ v(*x)
epsilon = 1e-4
alpha = beta = 0.5
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

# Zadanie 1 - Metoda Newtona z tłumieniem - wykres

X1 = np.linspace(-3, 2.5, 100)
X2 = np.linspace(-2, 2, 100)
x1, x2 = np.meshgrid(X1, X2)
f = np.exp(x1 + 3 * x2 - 0.1) + np.exp(-x1 - 0.1) + 1/8 * ((-1 + x1) * (7 * (-1 + x1) + np.sqrt(3) * (-1 + x2)) + (np.sqrt(3) * (-1 + x1) + 5 * (-1 + x2)) * (-1 + x2))
a = plt.contour(x1, x2, f, [2.5, 3.62, 5.34, 7.59, 19, 50, 200, 600], colors='k')
plt.scatter(x_path[:, 0], x_path[:, 1])
plt.clabel(a, fontsize=9, inline=True)
plt.title("Zadanie 1 - Metoda Newtona z tłumieniem")
plt.xlabel("$x1$")
plt.ylabel("$x2$")
plt.show()
