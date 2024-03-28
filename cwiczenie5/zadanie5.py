import numpy as np
import matplotlib.pyplot as plt

# Pierwsza funkcja
def phi1(x_):
    return 20 * x_ * x_ - 44 * x_ + 29

def dphi1(x_):
    return 40 * x_ - 44

def phi1_tangent(x_, x0=0):
    return phi1(x0) + alpha * x_ * dphi1(x0)

# Metoda backtrack search dla pierwszej funkcji
x = np.linspace(0, 2.5, 1000)
beta = 0.9
colors = ['r', 'g', 'b', 'c', 'y', 'm']
plt.figure(figsize=(12,8))
plt.plot(x, phi1(x), label="$\phi(s)=20s^2−44s+29$", c='k')
for idx, alpha in enumerate(np.linspace(0, 0.6, 6)):
    s = 2
    while phi1(s) >= phi1_tangent(s):
        s *= beta

    plt.scatter([s, s], [phi1(s), phi1_tangent(s)], color=colors[idx])
    plt.plot(x, phi1(0) + alpha * dphi1(0) * x, color=colors[idx], label=f"$y(s)=\phi(0) +{alpha}\phi'(0)s$")
plt.grid()
plt.title("Backtracking search method for $\phi(s) = 20s^2−44s+29$")
plt.xlabel("$s$")
plt.ylabel("$\phi(s), y(s)$")
plt.legend()
plt.ylim((0, 30))
plt.show()

# Druga funkcja
def phi2(x_):
    return 40 * x_ * x_ * x_ + 20 * x_ * x_ - 44 * x_ + 29

def dphi2(x_):
    return 120 * x_ * x_ + 40 * x_ - 44

def phi2_tangent(x_, x0=0):
    return phi2(x0) + alpha * x_ * dphi2(x0)


# Metoda backtrack search dla drugiej funkcji
x = np.linspace(0, 2.5, 1000)
beta = 0.9
colors = ['r', 'g', 'b', 'c', 'y', 'm']
plt.figure(figsize=(12,8))
plt.plot(x, phi2(x), label="$\phi(s)=40s^3+20s^2−44s+29$", c='k')
for idx, alpha in enumerate(np.linspace(0, 0.6, 6)):
    s = 2
    while phi2(s) >= phi2_tangent(s):
        s *= beta

    plt.scatter([s, s], [phi2(s), phi2_tangent(s)], color=colors[idx])
    plt.plot(x, phi2(0) + alpha * dphi2(0) * x, color=colors[idx], label=f"$y(s)=\phi(0) +{alpha}\phi'(0)s$")
plt.grid()
plt.title("Backtracking search method for $\phi(s) = 40s^3+20s^2−44s+29$")
plt.xlabel("$s$")
plt.ylabel("$\phi(s), y(s)$")
plt.legend()
plt.ylim((0, 30))
plt.show()
