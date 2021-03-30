import numpy as np
from matplotlib import pyplot as plt
# Hamiltonian of system is H = 0.5x**2 + 0.5p**2
time_unit = 0.001


def move_in_time(x, p):
    dx = p
    dp = -x
    return (x + dx*time_unit, p + dp*time_unit)

x_vec = np.array([0])
p_vec = np.array([1])

for i in range(10000):
    (next_x, next_p) = move_in_time(x_vec[len(x_vec) - 1], p_vec[len(p_vec) - 1])
    x_vec = np.append(x_vec, next_x)
    p_vec = np.append(p_vec, next_p)

t = np.linspace(0, 10000*0.001, 10001)
plt.plot(t, x_vec, "g", t, p_vec, "r")
plt.title("Displacement and Momentum in time")
plt.legend(["x", "p"])
plt.show()

