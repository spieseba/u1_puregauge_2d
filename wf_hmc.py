import gpt as g
import numpy as np

class u1_wilsonflow_hmc:
    @g.params_convention(step_size_lf=0.1, n_steps_lf=6)
    def __init__(self, rng, params):
        self.rng = rng
        self.params = params

    def __call__(self, V, beta, eps_wf, n_wf, mask, mask_rb):
        self.verbose = False
        self.Nd = len(V)
        eps_lf = self.params["step_size_lf"]
        n_lf = self.params["n_steps_lf"]

        t = g.timer("wilson-flowed hmc")
    
        t("lattice")
        P = [g.lattice(V[mu]) for mu in range(self.Nd)]

        t("random")
        self.rng.normal(P)

        t("hamiltonian")
        hamiltonian = 0.5 * sum([ g.sum( P[mu] * P[mu] ) for mu in range(self.Nd) ]) + S(V, beta, eps_wf, n_wf, mask, mask_rb)

        t("leapfrog")
        V_prime, P_prime = self.leapfrog(V, P, beta, eps_lf, n_lf, eps_wf, n_wf, mask, mask_rb)

        t("update")
        hamiltonian_prime = 0.5 * sum([ g.sum( P_prime[mu] * P_prime[mu] ) for mu in range(self.Nd) ]) + S(V_prime, beta, eps_wf, n_wf, mask, mask_rb)
        assert abs(hamiltonian_prime.imag) < 1e-15 # verify that hamiltonian is real
        dp = np.exp(hamiltonian - hamiltonian_prime.real)
        rn = self.rng.uniform_real()
        accept = dp > rn
        if accept: 
            for mu in range(self.Nd):
                V[mu] @= V_prime[mu]

        t()

    def leapfrog(self, V, P, beta, eps_lf, n_lf, eps_wf, n_wf, mask, mask_rb):
        V_prime = [g.lattice(V[mu]) for mu in range(self.Nd)]
        P_prime = [g.lattice(P[mu]) for mu in range(self.Nd)]
        for mu in range(self.Nd):
            V_prime[mu] @= V[mu]
            P_prime[mu] @= P[mu] 
        # initial step
        self.P_update(V_prime, P_prime, beta, 0.5 * eps_lf, eps_wf, n_wf, mask, mask_rb)
        # intermediate steps
        for k in range(n_lf-1):
            self.V_update(V_prime, P_prime, eps_lf)
            self.P_update(V_prime, P_prime, beta, eps_lf, eps_wf, n_wf, mask, mask_rb)
        # final step 
        self.V_update(V_prime, P_prime, eps_lf)
        self.P_update(V_prime, P_prime, beta, 0.5 * eps_lf, eps_wf, n_wf, mask, mask_rb)

        return V_prime, P_prime

    def V_update(self, V_prime, P_prime, eps_lf):
        for mu in range(self.Nd):
            V_prime[mu] @= g.component.exp( g(1j * eps_lf * P_prime[mu]) ) * V_prime[mu] 

    def P_update(self, V_prime, P_prime, beta, eps_lf, eps_wf, n_wf, mask, mask_rb):
        U_arr = F_n_eps_arr(V_prime, eps_wf, n_wf, mask, mask_rb)
        F = grad_S(U_arr, beta, eps_wf, n_wf, mask, mask_rb)
        for mu in range(self.Nd):
            P_prime[mu] @= P_prime[mu] - eps_lf * F[mu]


# staple sum - helper function
# remember that g.qcd.gauge.staple is already a sum and adjoint compared to my thesis
def staple_sum(U, mu):
    st = g.lattice(U[0])
    st[:] = 0
    for nu in range(len(U)):
        if mu != nu:
            st += g.qcd.gauge.staple(U, mu, nu) 
    return st 

# wilsonflow generator 
def wilsonflow_generator(U, mu):
    st = staple_sum(U, mu)
    return g( 1j * ( U[mu] * g.adj(st) - st * g.adj(U[mu]) ) )

# define wilson action
def wilson(U, beta):
    Nd = len(U)
    plaq_sum = 0
    for nu in range(Nd):
        for mu in range(nu):
            plaq_sum += g.sum( U[mu] * g.cshift(U[nu], mu, 1) * g.adj(g.cshift(U[mu], nu, 1)) * g.adj(U[nu]) )
    return - beta * plaq_sum.real

# field transformation
def F_n_eps(V, eps, n, mask, mask_rb):
    Nd = len(V)
    U = g.copy(V)
    for k in range(n):
        for cb in [g.even, g.odd]:
            mask[:] = 0
            mask_rb.checkerboard(cb)
            g.set_checkerboard(mask, mask_rb)
            for mu in range(Nd):
                U[mu] *= g.component.exp(1j * eps * wilsonflow_generator(U,mu) * mask) 
    return U

def F_n_eps_arr(V, eps, n, mask, mask_rb):
    Nd = len(V)
    U = g.copy(V)
    U_arr = [g.copy(V)]
    for k in range(n):
        for cb in [g.even, g.odd]:
            mask[:] = 0
            mask_rb.checkerboard(cb)
            g.set_checkerboard(mask, mask_rb)
            for mu in range(Nd):
                U[mu] *= g.component.exp(1j * eps * wilsonflow_generator(U,mu) * mask) 
                U_arr.append(g.copy(U))
    return U_arr



# action
def log_det(V, eps, mask, mu): 
    sigma = int(not mu)
    plaq1 = V[mu] * g.cshift(V[sigma], mu, 1) * g.adj( g.cshift(V[mu], sigma, 1) ) * g.adj( V[sigma] )
    plaq2 = V[mu] * g.adj( g.cshift( g.cshift(V[sigma], mu, 1), sigma, -1) ) * g.adj( g.cshift(V[mu], sigma, -1) ) * g.cshift(V[sigma], sigma, -1) 
    M = g(plaq1 + plaq2)
    eye = g.identity(g.u1(V[mu].grid))
    return (g.component.log(eye - eps * g( M + g.adj(M) )) * mask)

def S(V, beta, eps, n, mask, mask_rb):
    Nd = len(V)
    log_det_sum = 0
    U = g.copy(V)
    for k in range(n):
        for cb in [g.even, g.odd]:
            mask[:] = 0
            mask_rb.checkerboard(cb)
            g.set_checkerboard(mask, mask_rb)
            for mu in range(Nd):    
                log_det_sum -= g.sum( log_det(U, eps, mask, mu) )
                # update field
                U[mu] *= g.component.exp(1j * eps * wilsonflow_generator(U,mu) * mask) 
    return log_det_sum + wilson(F_n_eps(V, eps, n, mask, mask_rb), beta)



# compute plaquettes of all U[mu]
def compute_plaquettes(U, mu):
    Nd = len(U); sigma = int(not mu)
    plaq1 = U[mu] * g.cshift(U[sigma], mu, 1) * g.adj( g.cshift(U[mu], sigma, 1) ) * g.adj( U[sigma] ) 
    plaq2 = U[mu] * g.adj( g.cshift( g.cshift(U[sigma], mu, 1), sigma, -1) ) * g.adj( g.cshift(U[mu], sigma, -1) ) * g.cshift(U[sigma], sigma, -1) 
    return (plaq1, plaq2)

# gradient of field dependent part of H
def grad_S(U_arr, beta, eps, n, mask, mask_rb):  
    Nd = len(U_arr[0])
    # step 1
    index = -1; U_int = U_arr[index]
    plaq1_int = []; plaq2_int = []
    for mu in range(Nd):
        plaq1, plaq2 = compute_plaquettes(U_int,mu)
        plaq1_int.append(plaq1); plaq2_int.append(plaq2)
    M_int = [g(plaq1_int[mu] + plaq2_int[mu]) for mu in range(Nd)]
    F = [ g( - 0.5 * beta * (1j) * g(M_int[mu] - g.adj(M_int[mu])) ) for mu in range(Nd) ]
    
    # step 2
    for k in reversed(range(n)):
        for cb in reversed([g.even, g.odd]):
            mask[:] = 0
            mask_rb.checkerboard(cb)
            g.set_checkerboard(mask, mask_rb)
            for mu in reversed(range(Nd)):
                index -= 1; U = U_arr[index]; nu = int(not mu)
                plaq1_hat, plaq2_hat = compute_plaquettes(U, mu)
                M_hat = g(plaq1_hat + plaq2_hat)
                c = (1j) * eps * g.component.inv(g.identity(g.u1(M_hat.grid)) - eps * g(M_hat + g.adj(M_hat))) 
                F_mu = g.copy(F[mu])
    
                # (𝑧,𝜌)=(𝑥,𝜇)
                F_tmp = F[mu] + ( eps * F_mu * (-1) * g(M_hat + g.adj(M_hat)) 
                                  + c * g(M_hat - g.adj(M_hat))   ) * mask
                F[mu] = F_tmp

                # (𝑧,𝜌)=(𝑥+𝑎𝜇,𝜈)
                F_tmp = g.cshift(F[nu], mu, 1) + ( eps * F_mu * (-1) * g(plaq1_hat + g.adj(plaq1_hat)) 
                                                   + c * g(plaq1_hat - g.adj(plaq1_hat)) ) * mask
                F[nu] = g.cshift(F_tmp, mu, -1)           
                        
                # (𝑧,𝜌)=(𝑥+𝑎𝜈,𝜇) 
                F_tmp = g.cshift(F[mu], nu, 1) + ( eps * F_mu * g(plaq1_hat + g.adj(plaq1_hat)) 
                                                   + c * (-1) * g(plaq1_hat - g.adj(plaq1_hat))  ) * mask
                F[mu] = g.cshift(F_tmp, nu, -1)
                            
                # (𝑧,𝜌)=(𝑥,𝜈)
                F_tmp = F[nu] + ( eps * F_mu * g(plaq1_hat + g.adj(plaq1_hat)) 
                                  + c * (-1) * g(plaq1_hat - g.adj(plaq1_hat)) ) * mask
                F[nu] = F_tmp
                                    
                # (𝑧,𝜌)=(𝑥−𝑎𝜈,𝜇) 
                F_tmp = g.cshift(F[mu], nu, -1) + ( eps * F_mu * g(plaq2_hat + g.adj(plaq2_hat)) 
                                                    + c * (-1) * g(plaq2_hat - g.adj(plaq2_hat)) ) * mask
                F[mu] = g.cshift(F_tmp, nu, 1)
                            
                # (𝑧,𝜌)=(𝑥+𝑎𝜇−𝑎𝜈,𝜈) 
                F_tmp = g.cshift(g.cshift(F[nu], mu, 1), nu, -1) + ( eps * F_mu * g(plaq2_hat + g.adj(plaq2_hat)) 
                                                                     + c * (-1) * g(plaq2_hat - g.adj(plaq2_hat)) ) * mask
                F[nu] = g.cshift(g.cshift(F_tmp, nu, 1), mu, -1)
                            
                # (𝑧,𝜌)=(𝑥−𝑎𝜈,𝜈) 
                F_tmp = g.cshift(F[nu], nu, -1) + ( eps * F_mu * (-1) * g(plaq2_hat + g.adj(plaq2_hat)) 
                                                    + c * g(plaq2_hat - g.adj(plaq2_hat)) ) * mask
                F[nu] = g.cshift(F_tmp, nu, 1)
                       
    return F