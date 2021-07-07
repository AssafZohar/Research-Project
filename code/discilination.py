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

    def system_energy(self):
        energy = 0
        number_of_levels = (-5 + numpy.math.sqrt(25 + 40 * self.num_of_nodes)) / 10
        for i in range(int(number_of_levels)):
            k = i+1
            r_index = (1 + i) * i * 5 // 2
            nodes_in_level = k * 5
            for j in range(nodes_in_level):
                # there are always two neighbours in the same level, we calculate the first one
                energy += 0.5 * (self.lattice_constant \
                     - numpy.math.sqrt((self.locations[r_index + j][0] - self.locations[(j+1) % nodes_in_level + r_index][0])**2 \
                    + (self.locations[r_index + j][1] - self.locations[(j+1) % nodes_in_level + r_index][1])**2)) ** 2
                energy += 0.5 * (self.lattice_constant \
                     - numpy.math.sqrt((self.locations[r_index + j][0] - self.locations[(j-1) % nodes_in_level + r_index][0])**2 \
                    + (self.locations[r_index + j][1] - self.locations[(j-1) % nodes_in_level + r_index][1])**2)) ** 2
                # if the node is a corner it has 3 neighbours in the next level, otherwise it only has two
                nodes_in_next_level = nodes_in_level + 5
                next_r_index = (1 + k) * k * 5 // 2
                if (k == int(number_of_levels)):
                    continue

                if (j % k == 0):
                    energy += 0.5 * (self.lattice_constant \
                     - numpy.math.sqrt((self.locations[r_index + j][0] - self.locations[((j * (k + 1)) // k) + next_r_index][0])**2 \
                    + (self.locations[r_index + j][1] - self.locations[((j * (k + 1)) // k) + next_r_index][1])**2)) ** 2
                    energy += 0.5 * (self.lattice_constant \
                     - numpy.math.sqrt((self.locations[r_index + j][0] - self.locations[((((j * (k + 1)) // k) + 1) % nodes_in_next_level) + next_r_index][0])**2 \
                    + (self.locations[r_index + j][1] - self.locations[((((j * (k + 1)) // k) + 1) % nodes_in_next_level) + next_r_index][1])**2)) ** 2
                    energy += 0.5 * (self.lattice_constant \
                     - numpy.math.sqrt((self.locations[r_index + j][0] - self.locations[((((j * (k + 1)) // k) - 1) % nodes_in_next_level) + next_r_index][0])**2 \
                    + (self.locations[r_index + j][1] - self.locations[((((j * (k + 1)) // k) - 1) % nodes_in_next_level) + next_r_index][1])**2)) ** 2
                else:
                    energy += 0.5 * (self.lattice_constant \
                     - numpy.math.sqrt((self.locations[r_index + j][0] - self.locations[((j * (k + 1)) // k) + next_r_index][0])**2 \
                    + (self.locations[r_index + j][1] - self.locations[((j * (k + 1)) // k) + next_r_index][1])**2)) ** 2
                    energy += 0.5 * (self.lattice_constant \
                     - numpy.math.sqrt((self.locations[r_index + j][0] - self.locations[((((j * (k + 1)) // k) + 1) % nodes_in_next_level) + next_r_index][0])**2 \
                    + (self.locations[r_index + j][1] - self.locations[((((j * (k + 1)) // k) + 1) % nodes_in_next_level) + next_r_index][1])**2)) ** 2

        return energy

                    
                

if __name__ == "__main__":
    c = (10 + (10-1) * 5) * 5
    lattice = DiscilinatedLattice(15)
    lattice.set_locations()
    x_vector = lattice.locations[:,0]
    y_vector = lattice.locations[:,1]
    print(lattice.system_energy())
    