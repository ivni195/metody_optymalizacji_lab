import numpy as np
import cvxpy as cp

x_lek1 = cp.Variable()
x_lek2 = cp.Variable()
x_sur1 = cp.Variable()
x_sur2 = cp.Variable()

cost = 100 * x_sur1 + 199.90 * x_sur2 + 700 * x_lek1 + 800 * x_lek2
income = 6500 * x_lek1 + 7100 * x_lek2

objective = cp.Minimize(cost - income)

constraints = [ 
    0.01 * x_sur1 + 0.02 * x_sur2 - 0.5 * x_lek1 - 0.6 * x_lek2 >= 0,                # bilans
    x_sur1 + x_sur2 <= 1000,                                                         # ograniczenia zasobów magazynowych
    90 * x_lek1 + 100 * x_lek2 <= 2000,                                              # ograniczenia zasobów ludzkich
    40.00 * x_lek1 + 50.00 * x_lek2 <= 800,                                          # ograniczenia zasobów sprzętowych
    100.00 * x_sur1 + 199.90 * x_sur2 + 700.00 * x_lek1 + 800.00 * x_lek2 <= 100000, # ograniczenia budżetowe
    x_lek1 >= 0,
    x_lek2 >= 0,
    x_sur1 >= 0,
    x_sur2 >= 0,
]

p1 = cp.Problem(objective, constraints)
p1.solve(solver=cp.ECOS)

print(f"x_lek1 = {x_lek1.value:.3f}, x_lek2 = {x_lek2.value:.3f}, x_sur1 = {x_sur1.value:.3f}, x_sur2 = {x_sur2.value:.3f}")