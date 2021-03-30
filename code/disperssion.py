import numpy as np
from matplotlib import pyplot as plt

def get_relation(D):
    a = np.linspace(-9,10,20)
    x, y = np.meshgrid(a,a)

    # k_x values are only dependent on first reciprocal lattice vector
    k_x = 2 * np.math.pi * x / 20

    # k_y values are dependent of both reciproval lattice vectors
    k_y = - 2 * np.math.pi * x / 20 / np.math.sqrt(3) + 4 * np.math.pi * y / 20 / np.math.sqrt(3)

    # This is actually w^2
    w = (D * (6 - 2 * np.cos(k_x) - 4 * np.cos(k_x / 2) * np.cos(np.math.sqrt(3) * k_y / 2)))
    return np.sort(np.ndarray.flatten(w))

def main():
    w = get_relation(1)
    plt.plot(w,'ro')
    plt.show()

if __name__ == "__main__":
    main()