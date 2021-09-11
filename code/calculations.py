from discilination import DiscilinatedLattice
from centeredTriangular import TriangularCenteredLattice
from helperFunctions import harmonic_potential, IPR
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

def plot_multiple_lattices():
    number_of_nodes_disclinated = 0
    number_of_nodes_regular = 1
    for i in range(1, 20):
        number_of_nodes_disclinated += i*5
        number_of_nodes_regular += i*6
        if (i < 15):
            continue
        
        plot_triangular(number_of_nodes_regular)
        plot_disc(number_of_nodes_disclinated)

def plot_triangular(number_of_nodes_regular):
    lattice = TriangularCenteredLattice(number_of_nodes_regular)
    lattice.set_locations()
    a = numpy.ndarray.flatten(lattice.locations)
    result = minimize(lattice.energy_for_optimization, a)
    if not result.success:
        print("minimize energy failed for triangular lattice with {} nodes".format(number_of_nodes_regular))
        return
    
    lattice.set_dynamic_matrix()
    w, vec = lattice.find_eigen_values()
    plt.figure(clear=True)
    plt.plot(w, '.')
    plt.title("triangular lattice frequencies for N={} nodes".format(number_of_nodes_regular))
    plt.xlabel("Frequency number")
    plt.ylabel(r"$\omega$")
    plt.savefig(r"test\triangular_plot_{}".format(number_of_nodes_regular))        

def plot_disc(number_of_nodes_regular):
    lattice= DiscilinatedLattice(number_of_nodes_regular)
    lattice.set_locations()
    a = numpy.ndarray.flatten(lattice.locations)
    result = minimize(lattice.energy_for_optimization, a)
    if not result.success:
        print("minimize energy failed for Discilinated lattice with {} nodes".format(number_of_nodes_regular))
        return
    
    lattice.set_dynamic_matrix()
    w, vec = lattice.find_eigen_values()
    plt.figure(clear=True)
    plt.plot(w, '.')
    plt.title("triangular lattice frequencies for N={} nodes".format(number_of_nodes_regular))
    plt.xlabel("Frequency number")
    plt.ylabel(r"$\omega$")
    plt.savefig(r"test\discilinated_plot_{}.png".format(number_of_nodes_regular))       

def calculate_dynamics(t_max):
    c = (10 + (10-1) * 5) * 5
    lattice = DiscilinatedLattice(c)
    lattice.locations = numpy.loadtxt("flat_locations.csv", delimiter=',')
    lattice.set_dynamic_matrix([270, 269, 270 - 50])
    w, vec = lattice.find_eigen_values()
    used_values = numpy.array([-1])
    a = []
    for value in w:
        a += lattice.mode_spatial_by_frequency(value + 10**-13, w, used_values)
        numpy.append(used_values, value)

    print(len(a))
    
    for val in a:
        if val[0] < 0:
            print(val[0])
            val[0] = 0
        ax = plt.subplot()
        X = lattice.locations[:,0]
        Y = lattice.locations[:,1]
        cmhot = plt.get_cmap("coolwarm")
        im = ax.scatter(X, Y, c=val[1], cmap=cmhot, s=100)
        plt.colorbar(im, ax=ax)
        plt.title("Mode spatial distribution for $\omega$={0:9.2f}".format(numpy.sqrt(val[0])), fontsize=18)
        plt.xlabel("X (arbitrary)", fontsize=17)
        plt.ylabel("Y (arbitrary)", fontsize=17)
        filename = r"..\plots\modes\discilinated_locations2w={0:9.5f}{1:0.3f}.png".format(numpy.sqrt(val[0]), numpy.random.sample())
        plt.savefig(filename)
        plt.close()
    
    return
    X = lattice.locations[:,0]
    Y = lattice.locations[:,1]
    test_vector = numpy.zeros((275,))
    test_vector = gaussian(lattice.locations[:,0], lattice.locations[:,1])
    test_vector[271] = 0
    test_vector[269] = 0
    test_vector[272] = - 0.3
    test_vector[268] = - 0.3
    test_vector[270 - 49] = -test_vector[270 - 49]
    test_vector[270 - 50] = -test_vector[270 - 50]
    test_vector[270 - 48] = 0
    test_vector[270 - 51] = 0
    test_vector = test_vector / numpy.linalg.norm(test_vector)
    ax = plt.subplot()
    cmhot = plt.get_cmap("coolwarm")
    im = ax.scatter(X, Y, c=test_vector, cmap=cmhot, s=100)
    plt.colorbar(im, ax=ax)
    plt.show()

    times = numpy.linspace(0, t_max, 500)
    
    ipr = []
    filenames = []
    for t in times:
        sum = numpy.zeros((275,))
        # velocity = numpy.zeros((275,))
        for val in a:
            sum += (val[1] @ test_vector) * val[1] * numpy.cos(numpy.sqrt(val[0]) * t)
            # velocity += (val[1] @ test_vector) * val[1] * numpy.sqrt(val[0]) * numpy.sin(numpy.sqrt(val[0]) * t)
        # potential_energy = lattice.potential_energy(sum)
        ipr.append(IPR(sum))
        
        Z = sum
        ax = plt.subplot()
        cmhot = plt.get_cmap("coolwarm")
        im = ax.scatter(X, Y, c=Z, cmap=cmhot, s=100)
        plt.colorbar(im, ax=ax)
        plt.title("Mode spatial distribution for t={}".format(t))
        filename = r"..\plots\dynamics\discilinated_energy_t={}.png".format(t)
        
        filenames.append(filename)
        
        plt.savefig(filename)
        plt.close()
    
    with imageio.get_writer('..\\plots\\gaussian_mode_outer13.mp4', fps=10) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    plt.plot(times, ipr)
    plt.title("IPR over time")
    plt.xlabel("time (arbitrary)")
    plt.ylabel("IPR")
    plt.show()

def gaussian(x,y):
    x_mean = 6.4
    y_mean = -4.6
    return numpy.exp(-((x - x_mean) **2 + (y - y_mean)**2)/4)
    

def main():
    c = (10 + (10-1) * 5) * 5
    c2 = (5+10+15+20+25 +30 +35)
    c3 = 1 + (12 + (10-1)*6)*5
    """
    lattice = TriangularCenteredLattice(c3)
    #lattice.locations = numpy.loadtxt("flat_locations.csv", delimiter=',')
    lattice.set_locations()
    a = numpy.ndarray.flatten(lattice.locations)
    result = minimize(lattice.energy_for_optimization, a)
    print(result.success)
    lattice.set_dynamic_matrix()
    w, vec = lattice.find_eigen_values()
    print(w)
    w[0] = 0
    w_for_plot = numpy.sqrt(w)
    """
    #print(numpy.median(w_for_plot))
    lattice = DiscilinatedLattice(c)
    lattice.locations = numpy.loadtxt("flat_locations.csv", delimiter=',')
    lattice.set_dynamic_matrix()
    w, vec = lattice.find_eigen_values()
    w_for_plot = numpy.sqrt(w)
    
    
    ax = sns.distplot(w_for_plot, kde=True, rug=True, hist=False, vertical=True, kde_kws={"gridsize": 500, "cut":5, "bw": 0.2})
    ax.set_xticklabels(ax.get_xticks(), size = 12)
    plt.locator_params(axis='x', nbins=3)
    ax.set_yticklabels(ax.get_yticks(), size = 12)
    plt.title('Density of states discilinated lattice', fontsize=20)
    plt.ylabel(r"$\omega$ $\left[\left(\frac{\varepsilon}{m\sigma ^ 2} \right) ^ \frac{1}{2} \right]$", fontsize=20)
    plt.xlabel('Density', fontsize=20)
    plt.show()
    
    


if __name__ == "__main__":
    main()
    #plot_multiple_lattices()
    #calculate_dynamics(20)

    

    """
    c = (10 + (10-1) * 5) * 5
    c2 = (5+10+15+20+25 +30 +35)
    c3 = 1 + (12 + (10-1)*6)*5
    
    lattice = DiscilinatedLattice(c)
    lattice.locations = numpy.loadtxt("flat_locations.csv", delimiter=',')
    #a = numpy.ndarray.flatten(lattice.locations)
    #result = minimize(lattice.energy_for_optimization, a)
    #print(result.success)
    lattice.set_dynamic_matrix()
    w, vec = lattice.find_eigen_values()
    used_values = numpy.array([-1])
    a = []
    for value in w:
        a += lattice.find_vectors_and_repricosity(value + 10**-13, w, used_values)
        numpy.append(used_values, value)

    x= []
    y =[]
    used_values = numpy.array([-1])
    for b in a:
        if (b[2] < 20):
            c = lattice.find_vectors_and_repricosity2(b[0], w, used_values)
            vec = c[0][1]
            print(len(c))
            fig, ax = plt.subplots()
            cmhot = plt.get_cmap("coolwarm")
            im = ax.scatter(lattice.locations[:,0], lattice.locations[:,1], c=vec, cmap=cmhot, vmin=-1, vmax=1)
            ax.set_label("IPR={0:9.3f}".format(b[2]))
            plt.colorbar(im, ax=ax)
            plt.title("Mode spatial distribution for $\omega$={0:9.2f}".format(numpy.sqrt(b[0])), fontsize=18)
            plt.xlabel("X (arbitrary)", fontsize=17)
            plt.ylabel("Y (arbitrary)", fontsize=17)
            plt.savefig(r"..\plots\IPR\fdiscilinated_IPR_w={0:9.5f}.png".format(b[0]))  
    """     

    """
    print(len(x))
    fig, ax = plt.subplots()
    ax.plot(x,y, '.', markersize=8)
    line = mlines.Line2D([-1, 50], [15, 15], color='red')
    line.set_label("IPR = 15")
    ax.add_line(line)
    ax.set_xlim(-0.8, 50)
    plt.xlabel("$\omega$ (arbitrary)", fontsize=17)
    plt.ylabel("IPR", fontsize=17)
    plt.title("Discilinated lattice IPR by frequency", fontsize=18)
    ax.legend()
    plt.show()
    """
    
        
    #print(lattice.locations)
    #w, vec = lattice.find_eigen_values()
    #plot_density_of_states(w)
    """
    for i in range(1,10):
        print(i)
        lattice = DiscilinatedLattice(c, i)
        lattice.set_locations()

        a = numpy.ndarray.flatten(lattice.locations)
        result = minimize(lattice.energy_for_optimization, a)
        if not result.success:
            continue

        nodes = numpy.ndarray.reshape(result.x, (-1,2))

        lattice.locations = nodes # numpy.loadtxt("flat_locations.csv", delimiter=',')
        plt.figure(clear=True)
        plt.plot(nodes[:,0], nodes[:,1], '.')
        plt.title("triangular lattice locations for lattice const a={} ".format(lattice.lattice_constant))
        plt.savefig(r"..\plots\lattice_const\discilinated_locations_plot_{}.png".format(lattice.lattice_constant))  
        lattice.set_dynamic_matrix()
        w, vec = lattice.find_eigen_values()
        plt.figure(clear=True)
        plt.plot(w, '.')
        plt.title("triangular lattice frequencies for lattice const a={} ".format(lattice.lattice_constant))
        plt.xlabel("Frequency number")
        plt.ylabel(r"$\omega$")
        plt.savefig(r"..\plots\lattice_const\discilinated_plot_{}.png".format(lattice.lattice_constant))  
    used_values = numpy.array([-1])
    a = []
    for value in w:
        a += lattice.find_vectors_and_repricosity(value + 10**-13, w, used_values)
        numpy.append(used_values, value)

    numpy.savetxt("IPR.csv" ,numpy.array(a), delimiter=",")

    
    X = x_vector
    Y = y_vector
    Z = vec2[:,0]
    print(Z)
    # plt.contourf(X,Y,Z) plt.pcolormesh(X,Y,Z, shading="auto")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.scatter3D(X, Y, Z)
    plt.title("Mode spatial distribution for w={}".format(w2[0]))
    
    plt.show()

    #print(lattice.system_energy())"""
