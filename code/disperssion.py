import numpy as np
from matplotlib import pyplot as plt
from polygon import get_k_values, is_in_hexagon
from math import pi, sqrt

def get_relation(D, for_plot = True):
    k_x , k_y = get_k_values()
    x = is_in_hexagon(k_x, k_y)
    k_x = k_x[x]
    k_y = k_y[x]

    # This is actually w^2
    w = (D * (6 - 2 * np.cos(k_x) - 4 * np.cos(k_x / 2) * np.cos(np.math.sqrt(3) * k_y / 2)))
    if (for_plot):
        return np.sort(np.ndarray.flatten(w))
    else:
        return (k_x, k_y, w)

def main():
    x, y, z = get_relation(1, False)
    plt.plot(x,y, 'ro')
    a = [[2 * pi / 3, 2 * pi /sqrt(3)], [ 4 * pi / 3 , 0], [2 * pi / 3, - 2 * pi /sqrt(3)], [-2 * pi / 3, -2 * pi /sqrt(3)], [- 4 * pi / 3 , 0], [-2 * pi / 3,  2 * pi /sqrt(3)]]
    a.append(a[0])
    c,d = zip(*a)
    plt.plot(c,d)
    plt.show()

if __name__ == "__main__":
    main()