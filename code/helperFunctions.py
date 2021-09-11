import numpy

def harmonic_potential(x1, x2, y1, y2, sigma):
    epsilon = 1
    r = numpy.math.sqrt((x1 - x2)**2 + (y1-y2)**2)
    return 0.5* epsilon * (sigma - r) ** 2
    #return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

def IPR(vector):
    norm_vec = vector / numpy.linalg.norm(vector)
    sum = numpy.sum(numpy.power(norm_vec, 4))
    return 1/sum