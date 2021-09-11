from triangular_lattice import AbstractLattice
import numpy
import scipy
import scipy.sparse.linalg as linalg
import scipy.sparse as sparse
from matplotlib import pyplot as plt

class OneDLattice(AbstractLattice):
    def __init__(self, num_of_nodes):
        # Define primitive vectors and setup locations matrix
        super().__init__(num_of_nodes)
    
    def set_matrix(self, interaction_strength):
        # define dynamic matrix and set up a sparse dynamic matrix object for calculations
        for i in range(self.num_of_nodes):
            self.set_neighbour_interaction(i, i+1, -interaction_strength)
            self.set_neighbour_interaction(i, i-1, -interaction_strength)
            self.set_neighbour_interaction(i, i, 2*interaction_strength)
        self.sparse_dynamic_matrix = sparse.csr_matrix(self.dynamic_matrix)

    def get_analytic(self, D):
        # calculate the normal mode frequncies using momentum space considerations
        k_x = numpy.linspace( - self.num_of_nodes / 2 + 1, self.num_of_nodes/2, self.num_of_nodes)
        k = k_x * 2 * numpy.pi / self.num_of_nodes
        w = 2 * D * (1- numpy.cos(k))
        index = numpy.argsort(w)
        return w[index], k[index]
    
    def eigen_value_check(self, eigen_value, k_values):
        # helper function, receives omega and the negative and positive k values assosiated with it
        # checks whether the normal modes are spanned by a linear combination of sine and cosine functions with the apropriate k_values
        val, vec = linalg.eigsh(self.sparse_dynamic_matrix, 2, sigma=eigen_value)
        vec1 = vec[:,0]
        vec2 = vec[:,1]

        r1 = 1.72 * numpy.cos(- k_values[0] * numpy.linspace(0,399,400)) + numpy.sin(- k_values[0] * numpy.linspace(0,399,400))
        r1 = r1 / scipy.linalg.norm(r1)

        r2 = 1.72 * numpy.cos(- k_values[1] * numpy.linspace(0,399,400)) + numpy.sin(- k_values[1] * numpy.linspace(0,399,400))
        r2 = r2 / scipy.linalg.norm(r1)

        return abs(vec1 @ r1) **2 + abs(vec2 @ r1) **2 + abs(vec1 @ r2) **2 + abs(vec2 @ r2) **2


