import numpy as np
from matplotlib import pyplot as plt
from polygon import get_k_values, is_in_hexagon
from math import pi, sqrt

def get_relation(D, for_plot = True):
    k_x , k_y = get_k_values(28)
    x = is_in_hexagon(k_x, k_y)
    K_x = k_x[x]
    K_y = k_y[x]
    # This is actually w^2
    w = (D * (6 - 2 * np.cos(k_x) - 4 * np.cos(k_x / 2) * np.cos(np.math.sqrt(3) * k_y / 2)))
    w2 = (D * (6 - 2 * np.cos(K_x) - 4 * np.cos(K_x / 2) * np.cos(np.math.sqrt(3) * K_y / 2)))
    if (for_plot):
        return np.sort(np.ndarray.flatten(w2))
    else:
        return (k_x, k_y, w, K_x, K_y, w2)

def main():
    x, y, z, kx, ky, w2 = get_relation(1, False)
    fig, ax = plt.subplots()
    CS = ax.contour(x,y,z, [1,2,4,6,7,8,8.5,9])
    #ax.imshow((x,y,z))
    ax.clabel(CS, inline=True, fontsize=10)
    plt.plot(kx, ky, 'b.')
    a = [[2 * pi / 3, 2 * pi /sqrt(3)], [ 4 * pi / 3 , 0], [2 * pi / 3, - 2 * pi /sqrt(3)], [-2 * pi / 3, -2 * pi /sqrt(3)], [- 4 * pi / 3 , 0], [-2 * pi / 3,  2 * pi /sqrt(3)]]
    a.append(a[0])
    c,d = zip(*a)
    plt.plot(c,d)
    plt.show()

if __name__ == "__main__":
    main()