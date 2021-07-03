from triangular_lattice import AbstractLattice
from matplotlib import pyplot as plt
import numpy

class DiscilinatedLattice(AbstractLattice):
    def __init__(self, num_of_nodes):
        super().__init__(num_of_nodes)
        self.locations = numpy.zeros((num_of_nodes, 2))
        self.lattice_constant = 1

    def set_locations(self):
        number_of_levels = (-5 + numpy.math.sqrt(25 + 40 * self.num_of_nodes)) / 10
        if int(number_of_levels) != number_of_levels:
            raise("number of levels is not integer")

        for i in range(int(number_of_levels)):
            k = i+1
            r_index = (1 + i) * i * 5 // 2
            nodes_in_level = k * 5
            for j in range(nodes_in_level):
                self.locations[r_index + j][0] = k * numpy.math.cos(2 * numpy.math.pi * j / nodes_in_level)
                self.locations[r_index + j][1] = k * numpy.math.sin(2 * numpy.math.pi * j / nodes_in_level)

if __name__ == "__main__":
    lattice = DiscilinatedLattice((10 + (10-1) * 5) * 5)
    lattice.set_locations()
    x_vector = lattice.locations[:,0]
    y_vector = lattice.locations[:,1]
    plt.plot(x_vector, y_vector, '.')
    plt.show()
    