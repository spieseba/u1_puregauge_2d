import gpt as g
from metropolis import u1_metropolis
from observables import topological_charge

# grid 
L = [16,16]
grid = g.grid(L, g.single)
grid_eo = g.grid(L, g.single, g.redblack)
Nd = len(L)

# red/black mask
mask_rb = g.complex(grid_eo)
mask_rb[:] = 1

# full mask
mask = g.complex(grid)

# initialize rng
g.default.push_verbose("random", False)
rng = g.random("test", "vectorized_ranlux24_24_64")

# initialize field
U = [g.u1(grid) for mu in range(Nd)]
# cold start
for mu in range(Nd):
    U[mu][:] = 1

# simple plaquette action
def staple(U, mu):
    st = g.lattice(U[0])
    st[:] = 0
    for nu in range(len(U)):
        if mu != nu:
            st += g.qcd.gauge.staple(U, mu, nu) 
    return st

# metropolis sweeps
beta = 3.0
markov = u1_metropolis(rng)
for it in range(10):
    plaq = g.qcd.gauge.plaquette(U)
    Qtop = topological_charge(U)
    g.message(f"U(1) Metropolis {it+1} has P = {plaq} and Q_top {Qtop:.6f}")
    for cb in [g.even, g.odd]:
        mask[:] = 0
        mask_rb.checkerboard(cb)
        g.set_checkerboard(mask, mask_rb)
        for mu in range(Nd):
            st = g.eval(beta * staple(U, mu))
            markov(U[mu], st, mask)