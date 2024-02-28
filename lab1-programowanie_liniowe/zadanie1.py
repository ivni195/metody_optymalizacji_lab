import numpy as np
import cvxpy as cp

x1 = cp.Variable() # przenica
x2 = cp.Variable() # soja
x3 = cp.Variable() # maczka

objective = cp.Minimize(300 * x1 + 500 * x2 + 800 * x3)
constraints = [ 
    0.8 * x1 + 0.3 * x2 + 0.1 * x3 >= 0.3,  # weglowodany
    0.01 * x1 + 0.4 * x2 + 0.7 * x3 >= 0.7, # bialko
    0.15 * x1 + 0.1 * x2 + 0.2 * x3 >= 0.1, # sole mineralne
    x1 >= 0,
    x2 >= 0,
    x3 >= 0,
]
p1 = cp.Problem(objective, constraints)
p1.solve(solver=cp.ECOS)

print(f"x1 = {x1.value:.4f}, x2 = {x2.value:.4f}, x3 = {x3.value:.4f}")