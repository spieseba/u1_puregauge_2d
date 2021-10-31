import gpt as g
import numpy as np

class u1_hmc:
    @g.params_convention(step_size=0.1, n_steps=5)
    def __init__(self, rng, params):
        self.rng = rng
        self.params = params

    def __call__(self, U, beta):
        self.verbose = False
        step_size = self.params["step_size"]
        n_steps = self.params["n_steps"]

        t = g.timer("hmc")
    
        t("action")
        action = - beta * plaquette_sum(U).real

        t("lattice")
        Nd = len(U)
        P = [g.lattice(U[mu]) for mu in range(Nd)]

        t("random")
        self.rng.normal(P)

        t("hamiltonian")
        hamiltonian = action + 0.5 * sum([ g.sum( P[mu] * P[mu] ) for mu in range(Nd) ])

        t("leapfrog")
        U_prime, P_prime = self.leapfrog(U, P, beta, step_size, n_steps)

        t("update")
        action_prime =  - beta * plaquette_sum(U_prime).real
        hamiltonian_prime = action_prime + 0.5 * sum([ g.sum( P_prime[mu] * P_prime[mu] ) for mu in range(Nd) ])
        dp = np.exp(hamiltonian - hamiltonian_prime)
        rn = self.rng.uniform_real()
        accept = dp > rn
        if accept: 
            for mu in range(Nd):
                U[mu] @= U_prime[mu]

        t()
            

    def leapfrog(self, U, P, beta, step_size, n_steps):
        Nd = len(U)
        U_prime = [g.lattice(U[mu]) for mu in range(Nd)]
        P_prime = [g.lattice(P[mu]) for mu in range(Nd)]
        for mu in range(Nd):
            U_prime[mu] @= U[mu]
            P_prime[mu] @= P[mu] 
        # initial step
        self.P_update(U_prime, P_prime, beta, 0.5 * step_size)
        # intermediate steps
        for k in range(n_steps-1):
            self.U_update(U_prime, P_prime, step_size)
            self.P_update(U_prime, P_prime, beta, step_size)
        # final step 
        self.U_update(U_prime, P_prime, step_size)
        self.P_update(U_prime, P_prime, beta, 0.5 * step_size) 

        return U_prime, P_prime

    def U_update(self, U_prime, P_prime, step_size):
        Nd = len(U_prime)
        for mu in range(Nd):
            U_prime[mu] @= g.component.exp( g(1j * step_size * P_prime[mu]) ) * U_prime[mu] 

    def P_update(self, U_prime, P_prime, beta, step_size):
        Nd = len(P_prime)
        for mu in range(Nd):
            P_prime[mu] @= P_prime[mu] - step_size * self.force(U_prime[mu], staple(U_prime,mu), beta) 

    def force(self, link, staple, beta):
        f = staple
        f @= 0.5 * beta * g( 1j * ( g.adj(link * g.adj(f) ) - link * g.adj(f) ) )
        return f
    


# compute sum over all counter clockwise oriented plaquettes
def plaquette_sum(U):
    Nd = len(U)
    plaq = 0
    for nu in range(Nd):
        for mu in range(nu):
            plaq += g.sum( U[mu] * g.cshift(U[nu], mu, 1) * g.adj(g.cshift(U[mu], nu, 1)) * g.adj(U[nu]) )
    return plaq

def staple(U, mu):
    st = g.lattice(U[0])
    st[:] = 0
    for nu in range(len(U)):
        if mu != nu:
            st += g.qcd.gauge.staple(U, mu, nu) 
    return st

