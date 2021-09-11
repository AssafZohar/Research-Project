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
from helperFunctions import harmonic_potential, IPR

class DiscilinatedLattice(AbstractLattice):
    def __init__(self, num_of_nodes, lattice_constant = 1):
        super().__init__(num_of_nodes)
        self.locations = numpy.zeros((num_of_nodes, 2))
        self.lattice_constant = lattice_constant
        self.spring_constant = 1

    def set_locations(self):
        # hard coded for a discilinated lattice with an atom missing in the first layer
        number_of_levels = (-5 + numpy.math.sqrt(25 + 40 * self.num_of_nodes)) / 10
        if int(number_of_levels) != number_of_levels:
            raise("number of levels is not integer")

        for i in range(int(number_of_levels)):
            k = i+1
            r_index = (1 + i) * i * 5 // 2
            nodes_in_level = k * 5
            for j in range(nodes_in_level):
                self.locations[r_index + j][0] = self.lattice_constant * k * numpy.math.cos(2 * numpy.math.pi * j / nodes_in_level)
                self.locations[r_index + j][1] = self.lattice_constant * k * numpy.math.sin(2 * numpy.math.pi * j / nodes_in_level)

    def system_energy(self):
        energy = 0
        number_of_levels = (-5 + numpy.math.sqrt(25 + 40 * self.num_of_nodes)) / 10
        for i in range(int(number_of_levels)):
            k = i+1
            r_index = (1 + i) * i * 5 // 2
            nodes_in_level = k * 5
            for j in range(nodes_in_level):
                # there are always two neighbours in the same level, we calculate the first one
                node1 = self.locations[r_index + j]
                node2 = self.locations[(j+1) % nodes_in_level + r_index]
                energy += harmonic_potential(node1[0], node2[0], node1[1], node2[1], self.lattice_constant)
                # if the node is a corner it has 3 neighbours in the next level, otherwise it only has two
                nodes_in_next_level = nodes_in_level + 5
                next_r_index = (1 + k) * k * 5 // 2
                if (k == int(number_of_levels)):
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

    def lennard_jones_3d(self, x1, y1, z1, x2, y2, z2):
        r = numpy.math.sqrt((x1 - x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        return 4 * self.spring_constant * ((self.lattice_constant/r)**12 - (self.lattice_constant/r)**6)


    def potential_energy(self, z):
        energy = numpy.zeros((self.num_of_nodes,))
        number_of_levels = (-5 + numpy.math.sqrt(25 + 40 * self.num_of_nodes)) / 10
        for i in range(int(number_of_levels)):
            k = i+1
            r_index = (1 + i) * i * 5 // 2
            nodes_in_level = k * 5
            for j in range(nodes_in_level):
                # there are always two neighbours in the same level, we calculate the first one
                index1 = r_index + j
                node1 = self.locations[index1]
                index2 = (j+1) % nodes_in_level + r_index
                node2 = self.locations[index2]
                energy[index1] += self.lennard_jones_3d(node1[0], node1[1], z[index1], node2[0], node2[1], z[index2])
                energy[index2] += self.lennard_jones_3d(node1[0], node1[1], z[index1], node2[0], node2[1], z[index2])
                # if the node is a corner it has 3 neighbours in the next level, otherwise it only has two
                nodes_in_next_level = nodes_in_level + 5
                next_r_index = (1 + k) * k * 5 // 2
                if (k == int(number_of_levels)):
                    continue

                if (j % k == 0):
                    index2 = ((j * (k + 1)) // k) + next_r_index
                    node2 = self.locations[index2]
                    energy[index1] += self.lennard_jones_3d(node1[0], node1[1], z[index1], node2[0], node2[1], z[index2])
                    energy[index2] += self.lennard_jones_3d(node1[0], node1[1], z[index1], node2[0], node2[1], z[index2])
                    index2 = ((((j * (k + 1)) // k) + 1) % nodes_in_next_level) + next_r_index
                    node2 = self.locations[index2]
                    energy[index1] += self.lennard_jones_3d(node1[0], node1[1], z[index1], node2[0], node2[1], z[index2])
                    energy[index2] += self.lennard_jones_3d(node1[0], node1[1], z[index1], node2[0], node2[1], z[index2])
                    index2 = ((((j * (k + 1)) // k) - 1) % nodes_in_next_level) + next_r_index
                    node2 = self.locations[index2]
                    energy[index1] += self.lennard_jones_3d(node1[0], node1[1], z[index1], node2[0], node2[1], z[index2])
                    energy[index2] += self.lennard_jones_3d(node1[0], node1[1], z[index1], node2[0], node2[1], z[index2])
                    
                else:
                    index2 = ((j * (k + 1)) // k) + next_r_index
                    node2 = self.locations[index2]
                    energy[index1] += self.lennard_jones_3d(node1[0], node1[1], z[index1], node2[0], node2[1], z[index2])
                    energy[index2] += self.lennard_jones_3d(node1[0], node1[1], z[index1], node2[0], node2[1], z[index2])
                    index2 = ((((j * (k + 1)) // k) + 1) % nodes_in_next_level) + next_r_index
                    node2 = self.locations[index2]
                    energy[index1] += self.lennard_jones_3d(node1[0], node1[1], z[index1], node2[0], node2[1], z[index2])
                    energy[index2] += self.lennard_jones_3d(node1[0], node1[1], z[index1], node2[0], node2[1], z[index2])

        return energy

    def sparse_eigen_values(self, eigen_value, num_of_values):
        return linalg.eigsh(self.sparse_dynamic_matrix, num_of_values, sigma = eigen_value)

    def energy_for_optimization(self, flat_nodes):
        nodes = numpy.reshape(flat_nodes, (-1, 2))
        if (nodes.shape != self.locations.shape):
            raise "nodes array is {} while locations shape is {}".format(nodes.shape, self.locations.shape)

        self.locations = nodes
        return self.system_energy()

    def pre_strain(self, x1, x2, y1, y2):
        # second derivative of lennard jones potential by z where z=0
        return 4 * self.spring_constant * (6 * self.lattice_constant ** 6 / ((x1-x2)**2 + (y1-y2)**2) ** 4 - 12 * self.lattice_constant ** 12 / ((x1-x2)**2 + (y1-y2)**2) ** 7)

    def add_to_dynamic_matrix(self, i, j, zeroed_indices = []):
        # zeroed indices are used to remove nodes from the lattice
        if i in zeroed_indices or j in zeroed_indices:
            return
        
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


    def set_dynamic_matrix(self, zeroed_values = []):
        number_of_levels = (-5 + numpy.math.sqrt(25 + 40 * self.num_of_nodes)) / 10
        for i in range(int(number_of_levels)):
            k = i+1
            r_index = (1 + i) * i * 5 // 2
            nodes_in_level = k * 5
            for j in range(nodes_in_level):
                nodes_in_next_level = nodes_in_level + 5
                next_r_index = (1 + k) * k * 5 // 2
                # add both contributions to the matrix
                index1 = r_index + j
                if index1 in zeroed_values:
                    continue
                index2 = (j+1) % nodes_in_level + r_index
                self.add_to_dynamic_matrix(index1, index2, zeroed_values)
                
                if (k == int(number_of_levels)):
                    continue

                if (j % k == 0):
                    index2 = ((j * (k + 1)) // k) + next_r_index
                    self.add_to_dynamic_matrix(index1, index2, zeroed_values)
                    index2 = ((((j * (k + 1)) // k) + 1) % nodes_in_next_level) + next_r_index
                    self.add_to_dynamic_matrix(index1, index2, zeroed_values)
                    index2 = ((((j * (k + 1)) // k) - 1) % nodes_in_next_level) + next_r_index
                    self.add_to_dynamic_matrix(index1, index2, zeroed_values)

                else:
                    index2 = ((j * (k + 1)) // k) + next_r_index
                    self.add_to_dynamic_matrix(index1, index2, zeroed_values)

                    index2 = ((((j * (k + 1)) // k) + 1) % nodes_in_next_level) + next_r_index
                    self.add_to_dynamic_matrix(index1, index2, zeroed_values)

        self.sparse_dynamic_matrix = sparse.csr_matrix(self.dynamic_matrix)

    def IPR_by_frequency(self, eigen_value, eigen_values, used_values):
        new_vals = used_values[abs(used_values - eigen_value) < 10 ** -8]
        val_count = new_vals.shape[0]
        if (val_count != 0):
            return

        new_vals = eigen_values[abs(eigen_values - eigen_value) < 10 ** -8]
        val_count = new_vals.shape[0]
        # print(val_count)
        vals, vecs = self.sparse_eigen_values(eigen_value, val_count)
        a = []
        for i in range(vecs.shape[1]):
            a.append([vals[i], i, IPR(vecs[:,i])])
        return a

    def mode_spatial_by_frequency(self, eigen_value, eigen_values, used_values):
        new_vals = used_values[abs(used_values - eigen_value) < 10 ** -8]
        val_count = new_vals.shape[0]
        if (val_count != 0):
            return

        new_vals = eigen_values[abs(eigen_values - eigen_value) < 10 ** -8]
        val_count = new_vals.shape[0]
        #print((eigen_value, val_count))
        vals, vecs = self.sparse_eigen_values(eigen_value, val_count)
        a = []
        for i in range(vecs.shape[1]):
            a.append([vals[i], vecs[:,i]])
        return a









    