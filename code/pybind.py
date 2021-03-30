import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt

d = 1

pb.pltutils.use_style()

lattice = pb.Lattice(a1=[d,0], a2=[d/2, d*np.math.sqrt(3)/2])
lattice.add_sublattices(("A", [0,0]))
lattice.add_hoppings(([0,1], "A", "A", 1), ([1,0], "A", "A", 1))

model = pb.Model(lattice, pb.primitive(a1=5, a2=5))
model.plot()
plt.show()