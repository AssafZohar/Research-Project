from triangular_lattice import TriangularLattice
from matplotlib import pyplot as plt

def main():
    t_l = TriangularLattice(400)
    t_l.set_locations()
    t_l.set_matrix(1)
    # Finds 399 values in order to use sparse matrix methods
    a = t_l.find_eigen_values(400)[0]

    # Plotting
    w = t_l.get_analytic(1)
    plt.plot(w, 'b.')
    plt.plot(a, 'r.')
    plt.title("Eigen frequencies numeric and analytic")
    plt.ylabel("frequency (arbitrary)")
    plt.legend(["Analytic", "Numeric"])
    plt.show()


if __name__ == "__main__":
    main()