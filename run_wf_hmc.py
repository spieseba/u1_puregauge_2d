import gpt as g
from wf_hmc import u1_wilsonflow_hmc, F_n_eps
from observables import topological_charge

# grid 
L = [16,16]; Nd = len(L)
grid = g.grid(L, g.double)
grid_eo = g.grid(L, g.double, g.redblack)

# red/black mask
mask_rb = g.complex(grid_eo)
mask_rb[:] = 1

# full mask
mask = g.complex(grid)

# initialize rng
g.default.push_verbose("random", False)
rng = g.random("test", "vectorized_ranlux24_24_64")

# cold start
V = [g.u1(grid) for mu in range(Nd)]
for mu in range(Nd):
    V[mu][:] = 1

# wilson-flowed hmc sweeps
beta = 3.0
eps = 1.0/16; n = 1  # Lie-Euler integration

# hmc sweeps
markov = u1_wilsonflow_hmc(rng, step_size_lf=0.1, n_steps_lf=6)
for it in range(10):
    U = F_n_eps(V, eps, n, mask, mask_rb)
    plaq = g.qcd.gauge.plaquette(U)
    Qtop = topological_charge(U)
    g.message(f"U(1) HMC {it+1} has P = {plaq} and Q_top {Qtop:.5f}")
    markov(V, beta, eps, n, mask, mask_rb)