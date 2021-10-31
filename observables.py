import gpt as g
import numpy as np

# bosonic/geometric defintion of the topological charge
def topological_charge(U):
    # compute all counter-clockwise oriented plaquettes
    plaq = g.lattice(U[0])
    mu = 0; nu = 1
    plaq @= U[mu] * g.cshift(U[nu], mu, 1) * g.adj(g.cshift(U[mu], nu, 1)) * g.adj(U[nu]) 
    # compute plaquette angles and sum over them
    phi = g.component.log(plaq)
    Qtop = g.sum(phi).imag / (2.0*np.pi)
    return Qtop
