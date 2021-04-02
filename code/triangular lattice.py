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
            self.set_neighbour_interaction(i, i+1, -interaction_strength)
            self.set_neighbour_interaction(i, i+1+row_up, -interaction_strength)
            self.set_neighbour_interaction(i, i+row_up, -interaction_strength)
            self.set_neighbour_interaction(i, i-1, -interaction_strength)
            self.set_neighbour_interaction(i, i-1-row_up, -interaction_strength)
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


def main():
    t_l = TriangularLattice(400)
    t_l.set_locations()
    t_l.set_matrix(1)
    # Finds 399 values in order to use sparse matrix methods
    a = t_l.find_eigen_values(398)[0]

    # Plotting
    w = get_relation(1)
    plt.plot(w, 'b-')
    plt.plot(a, 'r-')
    plt.title("Eigen frequencies numeric and analytic")
    plt.ylabel("frequency (arbitrary)")
    plt.legend(["Analytic", "Numeric"])
    plt.show()
    c = w-a
    plt.close()
    plt.plot(c, 'g.')
    plt.title("Eigen frequencies numeric and analytic difference")
    plt.ylabel("frequency (arbitrary)")
    plt.show()

if __name__ == "__main__":
    main()
