import gpt as g
import numpy as np

class u1_metropolis:
    @g.params_convention(project_method="defect")
    def __init__(self, rng, params):
        self.rng = rng
        self.params = params

    def __call__(self, link, staple, mask):
        project_method = self.params["project_method"]
        t = g.timer("metropolis")

        t("action")
        action = g.component.real(g.eval(-g.trace(link * g.adj(staple)) * mask))

        t("lattice")
        V = g.lattice(link)
        V_eye = g.identity(link)

        t("random")
        self.rng.uniform_real(V,min=0,max=2.0*np.pi)
        V @= g.component.exp(g(V * 1j))
    
        t("update")
        V = g.where(mask, V, V_eye)

        link_prime = g.eval(V * link)
        action_prime = g.component.real(g.eval(-g.trace(link_prime * g.adj(staple)) * mask))

        dp = g.component.exp(g.eval(action - action_prime))

        rn = g.lattice(dp)

        t("random")
        self.rng.uniform_real(rn)

        t("random")
        accept = dp > rn
        accept *= mask

        link @= g.where(accept, link_prime, link)

        t()


