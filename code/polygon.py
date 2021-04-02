from matplotlib import pyplot as plt
import numpy as np
from math import pi, sqrt
  
def get_k_values():
    a = np.linspace(-9,10,28)
    x, y = np.meshgrid(a,a)
    k_x = 4 * np.math.pi * x / 20
    k_y = 4 * pi /sqrt(3) * y / 20
    return (k_x, k_y)

def is_in_hexagon(x, y):
    c1 = y <= 2 * pi / sqrt(3)
    c2 = y > - 2 * pi / sqrt(3)
    c3 = y <= -sqrt(3) * x + 4 * pi / sqrt(3)
    c4 = y > -sqrt(3) * x - 4 * pi / sqrt(3)
    c5 = y <= sqrt(3) * x + 4 * pi / sqrt(3)
    c6 = y > sqrt(3) * x - 4 * pi / sqrt(3)
    return (c1 & c2 & c3 & c4 & c5 & c6)

def main():
    k_x , k_y = get_k_values()
    x = is_in_hexagon(k_x, k_y)
    k_x = k_x[x]
    k_y = k_y[x]
    a = [[2 * pi / 3, 2 * pi /sqrt(3)], [ 4 * pi / 3 , 0], [2 * pi / 3, - 2 * pi /sqrt(3)], [-2 * pi / 3, -2 * pi /sqrt(3)], [- 4 * pi / 3 , 0], [-2 * pi / 3,  2 * pi /sqrt(3)]]
    a.append(a[0])
    x,y = zip(*a)
    plt.plot(x,y)
    plt.plot(k_x, k_y, 'ro')
    plt.show()

if __name__ == "__main__":
    main()
