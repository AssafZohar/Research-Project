import numpy
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from matplotlib import pyplot as plt

class AbstractLattice:
    def __init__(self, num_of_nodes):
        self.num_of_nodes = num_of_nodes
        self.dynamic_matrix = numpy.zeros((self.num_of_nodes, self.num_of_nodes))
        self.sparse_dynamic_matrix = None
    
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
        self.sparse_dynamic_matrix = sparse.csr_matrix(self.dynamic_matrix)
    
    def set_locations(self):
        # Calculates Locations of all lattice points in equilibrium
        row_up = int(numpy.math.sqrt(self.num_of_nodes))
        for i in range(self.num_of_nodes):
            self.locations[i,0] = (i % row_up) * self.primitive_vectors[0,0] + (i // row_up) * self.primitive_vectors[1,0]
            self.locations[i,1] = (i % row_up) * self.primitive_vectors[0,1] + (i // row_up) * self.primitive_vectors[1,1]

    def find_eigen_values(self, num_of_values):
        return linalg.eigsh(self.dynamic_matrix, num_of_values)
    
    def sparse_eigen_values(self, eigen_value, num_of_values):
        return linalg.eigsh(self.sparse_dynamic_matrix, num_of_values, sigma = eigen_value)
    
    def get_analytic(self, D):
        row = int(numpy.math.sqrt(self.num_of_nodes))
        a = numpy.linspace( - row / 2 + 1, row/2, row)
        x, y = numpy.meshgrid(a,a)

        # k_x values are only dependent on first reciprocal lattice vector
        k_x = 2 * numpy.math.pi * x / row

        # k_y values are dependent of both reciproval lattice vectors
        k_y = - 2 * numpy.math.pi * x / row / numpy.math.sqrt(3) + 4 * numpy.math.pi * y / row / numpy.math.sqrt(3)
        # This is actually w^2
        a = numpy.linspace(- 2 * numpy.math.pi, 2 * numpy.math.pi, 100)
        w = (D * (6 - 2 * numpy.cos(k_x) - 4 * numpy.cos(k_x / 2) * numpy.cos(numpy.math.sqrt(3) * k_y / 2)))
        w = numpy.array(w).flatten()
        k_x = numpy.array(k_x).flatten()
        k_y = numpy.array(k_y).flatten()
        index = numpy.argsort(w)
        return w[index], k_x[index], k_y[index]
    
    def find_vectors_and_repricosity(self, eigen_value, eigen_values, vector):
        # helper function - checks if input vector is spanned by an eigen space of normal modes with the same frequency
        new_vals = eigen_values[abs(eigen_values - eigen_value) < 10 ** -5]
        val_count = new_vals.shape[0]
        vals, vecs = self.sparse_eigen_values(eigen_value, val_count)
        sum = 0
        for i in range(val_count):
            sum += abs(vector @ vecs[:, i])**2
        return sum

def main():
    # checks if planar waves are spanned by lattice normal modes (which implies normal modes are spanned by planar waves)
    t_l = TriangularLattice(400)
    t_l.set_locations()
    t_l.set_matrix(1)
    w, k_x, k_y = t_l.get_analytic(1)
    w2, vec = t_l.find_eigen_values(400)
    print(w)
    line = numpy.linspace(0, 399, 400)
    
    r = numpy.sin(- (k_x[371] * ((line % 20) - (line // 20) / 2) + (k_y[371] * numpy.math.sqrt(3) / 2 * (line // 20) )))
    r = r / numpy.linalg.norm(r)
   
    count =0
    for i in range(w.shape[0]):
        r = numpy.cos(- (k_x[i] * ((line % 20) - (line // 20) / 2) + (k_y[i] * numpy.math.sqrt(3) / 2 * (line // 20) )))
        r = r / numpy.linalg.norm(r)
        a = t_l.find_vectors_and_repricosity(w[i], w, r)
        if (a < 0.98):
            print(i, a)
            count += 1
    print(count)



if __name__ == "__main__":
    main()
