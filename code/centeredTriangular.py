from numpy.lib.arraysetops import unique
from numpy.lib.function_base import append
import scipy
from triangular_lattice import AbstractLattice
from matplotlib import pyplot as plt
from matplotlib import lines as mlines
import numpy
from scipy.optimize import minimize
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import imageio
import seaborn as sns
from helperFunctions import harmonic_potential

class TriangularCenteredLattice(AbstractLattice):
    def __init__(self, num_of_nodes):
        super().__init__(num_of_nodes)
        self.locations = numpy.zeros((num_of_nodes, 2))
        self.lattice_constant = 1
        self.spring_constant = 1
        self.number_of_levels = (-3 + numpy.math.sqrt(9 + 12 * (self.num_of_nodes - 1))) / 6
        if int(self.number_of_levels) != self.number_of_levels:
            raise("number of levels is not integer")

    def set_locations(self):
        # hard coded for a tringular lattice with a center atom

        self.locations[0][0] = 0
        self.locations[0][1] = 0
        for i in range(int(self.number_of_levels)):
            k = i+1
            r_index = (1 + i) * i * 6 // 2 + 1
            nodes_in_level = k * 6
            for j in range(nodes_in_level):
                self.locations[r_index + j][0] = k * numpy.math.cos(2 * numpy.math.pi * j / nodes_in_level)
                self.locations[r_index + j][1] = k * numpy.math.sin(2 * numpy.math.pi * j / nodes_in_level)

    def system_energy(self):
        energy = 0
        
        # first level is treated specially
        for i in range(1,7):
            node1 = self.locations[0]
            node2 = self.locations[i]
            energy += harmonic_potential(node1[0], node2[0], node1[1], node2[1], self.lattice_constant)

        for i in range(int(self.number_of_levels)):
            k = i+1
            r_index = (1 + i) * i * 6 // 2 + 1
            nodes_in_level = k * 6
            
            for j in range(nodes_in_level):
                # there are always two neighbours in the same level, we calculate the first one
                node1 = self.locations[r_index + j]
                node2 = self.locations[(j+1) % nodes_in_level + r_index]
                energy += harmonic_potential(node1[0], node2[0], node1[1], node2[1], self.lattice_constant)
                # if the node is a corner it has 3 neighbours in the next level, otherwise it only has two
                nodes_in_next_level = nodes_in_level + 6
                next_r_index = (1 + k) * k * 6 // 2 + 1
                if (k == int(self.number_of_levels)):
                    continue

                if (j % k == 0):
                    node2 = self.locations[((j * (k + 1)) // k) + next_r_index]
                    energy += harmonic_potential(node1[0], node2[0], node1[1], node2[1], self.lattice_constant)
                    node2 = self.locations[((((j * (k + 1)) // k) + 1) % nodes_in_next_level) + next_r_index]
                    energy += harmonic_potential(node1[0], node2[0], node1[1], node2[1], self.lattice_constant)
                    node2 = self.locations[((((j * (k + 1)) // k) - 1) % nodes_in_next_level) + next_r_index]
                    energy += harmonic_potential(node1[0], node2[0], node1[1], node2[1], self.lattice_constant)
                    
                else:
                    node2 = self.locations[((j * (k + 1)) // k) + next_r_index]
                    energy += harmonic_potential(node1[0], node2[0], node1[1], node2[1], self.lattice_constant)
                    node2 = self.locations[((((j * (k + 1)) // k) + 1) % nodes_in_next_level) + next_r_index]
                    energy += harmonic_potential(node1[0], node2[0], node1[1], node2[1], self.lattice_constant)

        return energy


    def energy_for_optimization(self, flat_nodes):
        nodes = numpy.reshape(flat_nodes, (-1, 2))
        if (nodes.shape != self.locations.shape):
            raise "nodes array is {} while locations shape is {}".format(nodes.shape, self.locations.shape)

        self.locations = nodes
        return self.system_energy()

    def pre_strain(self, x1, x2, y1, y2):
        # needs recalculating
        return 4 * self.spring_constant * (6 * self.lattice_constant ** 6 / ((x1-x2)**2 + (y1-y2)**2) ** 4 - 12 * self.lattice_constant ** 12 / ((x1-x2)**2 + (y1-y2)**2) ** 7)

    def add_to_dynamic_matrix(self, i, j):
        self.dynamic_matrix[i, j] += self.pre_strain(
            self.locations[i][0],
            self.locations[j][0],
            self.locations[i][1],
            self.locations[j][1])

        self.dynamic_matrix[j, i] += self.pre_strain(
            self.locations[i][0],
            self.locations[j][0],
            self.locations[i][1],
            self.locations[j][1])

        self.dynamic_matrix[i, i] -= self.pre_strain(
            self.locations[i][0],
            self.locations[j][0],
            self.locations[i][1],
            self.locations[j][1])

        self.dynamic_matrix[j, j] -= self.pre_strain(
            self.locations[i][0],
            self.locations[j][0],
            self.locations[i][1],
            self.locations[j][1])


    def set_dynamic_matrix(self):
        for i in range(1,7):
            index1 = 0
            index2 = i
            self.add_to_dynamic_matrix(index1, index2)

        for i in range(int(self.number_of_levels)):
            k = i+1
            r_index = (1 + i) * i * 6 // 2 + 1
            nodes_in_level = k * 6
            for j in range(nodes_in_level):
                nodes_in_next_level = nodes_in_level + 5
                next_r_index = (1 + k) * k * 6 // 2 + 1
                # add both contributions to the matrix
                index1 = r_index + j
                index2 = (j+1) % nodes_in_level + r_index
                self.add_to_dynamic_matrix(index1, index2)
                
                if (k == int(self.number_of_levels)):
                    continue

                if (j % k == 0):
                    index2 = ((j * (k + 1)) // k) + next_r_index
                    self.add_to_dynamic_matrix(index1, index2)
                    index2 = ((((j * (k + 1)) // k) + 1) % nodes_in_next_level) + next_r_index
                    self.add_to_dynamic_matrix(index1, index2)
                    index2 = ((((j * (k + 1)) // k) - 1) % nodes_in_next_level) + next_r_index
                    self.add_to_dynamic_matrix(index1, index2)

                else:
                    index2 = ((j * (k + 1)) // k) + next_r_index
                    self.add_to_dynamic_matrix(index1, index2)

                    index2 = ((((j * (k + 1)) // k) + 1) % nodes_in_next_level) + next_r_index
                    self.add_to_dynamic_matrix(index1, index2)