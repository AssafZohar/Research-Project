from triangular_lattice import AbstractLattice
import numpy
from disperssion import get_relation
import scipy.sparse.linalg as linalg
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

    def get_analytic(self, D):
        k_x = numpy.linspace( - self.num_of_nodes / 2 + 1, self.num_of_nodes/2, self.num_of_nodes)
        k = k_x * 2 * numpy.pi / self.num_of_nodes
        return numpy.sort(2 * D * (1- numpy.cos(k))) 

class TwoDLattice(AbstractLattice):
    def __init__(self, num_of_nodes):
        # Define primitive vectors and setup locations matrix
        super().__init__(num_of_nodes)
    
    def set_matrix(self, interaction_strength):
        row_up = int(numpy.sqrt(self.num_of_nodes))
        for i in range(self.num_of_nodes):
            row, col = i // row_up, i % row_up
            self.set_neighbour_interaction(i, row * row_up + ((col + 1) % row_up), -interaction_strength)
            self.set_neighbour_interaction(i, row * row_up + ((col - 1) % row_up), -interaction_strength)
            self.set_neighbour_interaction(i, i+row_up, -interaction_strength)
            self.set_neighbour_interaction(i, i-row_up, -interaction_strength)
            self.set_neighbour_interaction(i, i, 4*interaction_strength)

    def get_analytic(self, D):
        row = int(numpy.math.sqrt(self.num_of_nodes))
        k = numpy.linspace( - row / 2 + 1, row/2, row)
        k_x = k * 2 * numpy.pi / row
        k_y = k_x
        K_x, K_y = numpy.meshgrid(k_x,k_y)
        w = 4 * D * (numpy.sin(K_x/2) ** 2 + numpy.sin(K_y/2) ** 2) 
        return numpy.sort(numpy.ndarray.flatten(w)) 

def array_has_value(arr, value):
    for a in arr:
        if (abs(a- value) < 10 ** -12):
            return True
    return False

def main():
    a = TwoDLattice(20 * 20)
    a.set_matrix(1)
    b = a.find_eigen_values()[0]
    c = a.get_analytic(1)
    plt.plot(c, 'r.')
    plt.plot(b, 'b.')
    plt.title("Eigen frequencies numeric and analytic")
    plt.ylabel("frequency (arbitrary)")
    plt.legend(["Analytic", "Numeric"])
    plt.show()
    print(b-c)


if __name__ == "__main__":
    main()
