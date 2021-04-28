import numpy
from disperssion import get_relation
import scipy.sparse.linalg as linalg
from matplotlib import pyplot as plt

class AbstractLattice:
    def __init__(self, num_of_nodes):
        self.num_of_nodes = num_of_nodes
        self.dynamic_matrix = numpy.zeros((self.num_of_nodes, self.num_of_nodes))
    
    def set_neighbour_interaction(self, i, j, interaction_strength):
        # Helper function to abstract index assignment
        self.dynamic_matrix[i % self.num_of_nodes, j % self.num_of_nodes] = interaction_strength
    
    def find_eigen_values(self):
        return numpy.linalg.eigh(self.dynamic_matrix)

class TriangularLattice(AbstractLattice):
    def __init__(self, num_of_nodes):
        # Define primitive vectors and setup locations matrix
        super().__init__(num_of_nodes)
        self.primitive_vectors = numpy.array([[1,0],[1/2, numpy.math.sqrt(3)/2]])
        self.locations = numpy.zeros((num_of_nodes, 2))

    def set_matrix(self, interaction_strength):
        # interaction_strength is in units of Force/mass
        row_up = int(numpy.math.sqrt(self.num_of_nodes))
        for i in range(self.num_of_nodes):
            # for triangular lattice in rows each node sees
            row, col = i // row_up, i % row_up
            self.set_neighbour_interaction(i, row * row_up + ((col + 1) % row_up), -interaction_strength)
            self.set_neighbour_interaction(i, row * row_up + ((col - 1) % row_up), -interaction_strength)
            self.set_neighbour_interaction(i, (row + 1) * row_up + ((col + 1) % row_up), -interaction_strength)
            self.set_neighbour_interaction(i, (row - 1) * row_up + ((col - 1) % row_up), -interaction_strength)
            self.set_neighbour_interaction(i, i+row_up, -interaction_strength)
            self.set_neighbour_interaction(i, i-row_up, -interaction_strength)
            self.set_neighbour_interaction(i, i, 6* interaction_strength)
    
    def set_locations(self):
        # Calculates Locations of all lattice points in equilibrium
        row_up = int(numpy.math.sqrt(self.num_of_nodes))
        for i in range(self.num_of_nodes):
            self.locations[i,0] = (i % row_up) * self.primitive_vectors[0,0] + (i // row_up) * self.primitive_vectors[1,0]
            self.locations[i,1] = (i % row_up) * self.primitive_vectors[0,1] + (i // row_up) * self.primitive_vectors[1,1]

    def find_eigen_values(self, num_of_values):
        return linalg.eigsh(self.dynamic_matrix, num_of_values)
    
    def get_analytic(self, D):
        row = int(numpy.math.sqrt(self.num_of_nodes))
        a = numpy.linspace( - row / 2 + 1, row/2, row)
        x, y = numpy.meshgrid(a,a)

        # k_x values are only dependent on first reciprocal lattice vector
        k_x = 2 * numpy.math.pi * x / row

        # k_y values are dependent of both reciproval lattice vectors
        k_y = - 2 * numpy.math.pi * x / row / numpy.math.sqrt(3) + 4 * numpy.math.pi * y / row / numpy.math.sqrt(3)
        # This is actually w^2
        w = (D * (6 - 2 * numpy.cos(k_x) - 4 * numpy.cos(k_x / 2) * numpy.cos(numpy.math.sqrt(3) * k_y / 2)))
        return numpy.sort(numpy.ndarray.flatten(w))

def array_has_value(arr, value):
    for a in arr:
        if (abs(a- value) < 10 ** -2):
            return True
    return False

def main():
    t_l = TriangularLattice(400)
    t_l.set_locations()
    t_l.set_matrix(1)
    # Finds 399 values in order to use sparse matrix methods
    z = numpy.array(t_l.find_eigen_values(400)[1][300])
    x = numpy.array(t_l.locations[:,0])
    y = numpy.array(t_l.locations[:,1])
    X = numpy.reshape(x, (20,20))
    Y = numpy.reshape(y, (20,20))
    Z = numpy.reshape(z, (20, 20))
    # plt.contourf(X,Y,Z) plt.pcolormesh(X,Y,Z, shading="auto")
    plt.contourf(X,Y,Z)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
