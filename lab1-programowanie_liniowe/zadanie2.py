import numpy as np
import cvxpy as cp

x1 = cp.Variable() # płatki
x2 = cp.Variable() # mleko
x3 = cp.Variable() # chleb

objective = cp.Minimize(0.15 * x1 + 0.25 * x2 + 0.05 * x3)

constraints = [ 
    70 * x1 + 121 * x2 + 65 * x3 <= 2250,    # kalorie
    70 * x1 + 121 * x2 + 65 * x3 >= 2000,
    107 * x1 + 500 * x2 + 0.0 * x3 <= 10000, # witaminy
    107 * x1 + 500 * x2 + 0.0 * x3 >= 5000,
    45 * x1 + 40 * x2 + 60 * x3 <= 1000,     # cukier
    0 <= x1,
    x1 <= 10,
    0 <= x2,
    x2 <= 10,
    0 <= x3,
    x3 <= 10
]

p1 = cp.Problem(objective, constraints)
p1.solve(solver=cp.ECOS)

print(f"x1 = {x1.value:.4f}, x2 = {x2.value:.4f}, x3 = {x3.value:.4f}")