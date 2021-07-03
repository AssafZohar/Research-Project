from triangular_lattice import AbstractLattice
import numpy
import scipy
from disperssion import get_relation
import scipy.sparse.linalg as linalg
import scipy.sparse as sparse
from matplotlib import pyplot as plt

class OneDLattice(AbstractLattice):
    def __init__(self, num_of_nodes):
        # Define primitive vectors and setup locations matrix
        super().__init__(num_of_nodes)
    
    def set_matrix(self, interaction_strength):
        for i in range(self.num_of_nodes):
            self.set_neighbour_interaction(i, i+1, -interaction_strength)
            self.set_neighbour_interaction(i, i-1, -interaction_strength)
            self.set_neighbour_interaction(i, i, 2*interaction_strength)
        self.sparse_dynamic_matrix = sparse.csr_matrix(self.dynamic_matrix)

    def get_analytic(self, D):
        k_x = numpy.linspace( - self.num_of_nodes / 2 + 1, self.num_of_nodes/2, self.num_of_nodes)
        k = k_x * 2 * numpy.pi / self.num_of_nodes
        w = 2 * D * (1- numpy.cos(k))
        index = numpy.argsort(w)
        return w[index], k[index]
    
    def eigen_value_check(self, eigen_value, k_values):
        val, vec = linalg.eigsh(self.sparse_dynamic_matrix, 2, sigma=eigen_value)
        vec1 = vec[:,0]
        vec2 = vec[:,1]

        r1 = 1.72 * numpy.cos(- k_values[0] * numpy.linspace(0,399,400)) + numpy.sin(- k_values[0] * numpy.linspace(0,399,400))
        r1 = r1 / scipy.linalg.norm(r1)

        r2 = 1.72 * numpy.cos(- k_values[1] * numpy.linspace(0,399,400)) + numpy.sin(- k_values[1] * numpy.linspace(0,399,400))
        r2 = r2 / scipy.linalg.norm(r1)

        return abs(vec1 @ r1) **2 + abs(vec2 @ r1) **2 + abs(vec1 @ r2) **2 + abs(vec2 @ r2) **2


class TwoDLattice(AbstractLattice):
    def __init__(self, num_of_nodes):
        # Define primitive vectors and setup locations matrix
        super().__init__(num_of_nodes)
        self.locations = numpy.array([i for i in range(400)])
    
    def set_matrix(self, interaction_strength):
        row_up = int(numpy.sqrt(self.num_of_nodes))
        for i in range(self.num_of_nodes):
            row, col = i // row_up, i % row_up
            self.set_neighbour_interaction(i, row * row_up + ((col + 1) % row_up), -interaction_strength)
            self.set_neighbour_interaction(i, row * row_up + ((col - 1) % row_up), -interaction_strength)
            self.set_neighbour_interaction(i, i+row_up, -interaction_strength)
            self.set_neighbour_interaction(i, i-row_up, -interaction_strength)
            self.set_neighbour_interaction(i, i, 4*interaction_strength)
        self.sparse_dynamic_matrix = sparse.csr_matrix(self.dynamic_matrix)

    def get_analytic(self, D):
        row = int(numpy.math.sqrt(self.num_of_nodes))
        k = numpy.linspace( - row / 2 + 1, row/2, row)
        k_x = k * 2 * numpy.pi / row
        k_y = k_x
        K_x, K_y = numpy.meshgrid(k_x,k_y)
        w = 4 * D * (numpy.sin(K_x/2) ** 2 + numpy.sin(K_y/2) ** 2) 
        k_x = numpy.array(K_x).flatten()
        k_y = numpy.array(K_y).flatten()
        w = numpy.array(w).flatten()
        index = numpy.argsort(w)
        return w[index], k_x[index], k_y[index]
    
    def sparse_eigen_values(self, eigen_value, num_of_values):
        return linalg.eigsh(self.sparse_dynamic_matrix, num_of_values, sigma = eigen_value)

def main():
    a = TwoDLattice(20 * 20)
    a.set_matrix(1)
    w, k_x, k_y = a.get_analytic(1)
    print(w)
    w2, vec = a.sparse_eigen_values(w[9], 4)
    v1, v2, v3, v4 = vec[:,0], vec[:,1], vec[:,2], vec[:,3]
    line = numpy.linspace(0, 399, 400)
    r = numpy.sin(- (k_x[9] * (line % 20) + k_y[9] * (line // 20)))
    r = r / numpy.linalg.norm(r)
    print(abs(r @ v1) ** 2 + abs(r @ v2) ** 2 + abs(r @ v3) ** 2 + abs(r @ v4) ** 2)
        
    
    
    
    
    
if __name__ == "__main__":
    main()
