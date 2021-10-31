import gpt as g
from hmc import u1_hmc
from observables import topological_charge

# grid 
L = [16,16]
grid = g.grid(L, g.double)
Nd = len(L)

# initialize rng
g.default.push_verbose("random", False)
rng = g.random("test", "vectorized_ranlux24_24_64")

# initialize field
U = [g.u1(grid) for mu in range(Nd)]
# cold start
for mu in range(Nd):
    U[mu][:] = 1

# hmc sweeps
beta = 3.0
markov = u1_hmc(rng, step_size=0.1, n_steps=6)
for it in range(10):
    plaq = g.qcd.gauge.plaquette(U)
    Qtop = topological_charge(U)
    g.message(f"U(1) HMC {it+1} has P = {plaq} and Q_top {Qtop:.5f}")
    markov(U, beta)
