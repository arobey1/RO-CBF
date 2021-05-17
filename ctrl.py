import cvxpy as cp
import jax
import jax.numpy as jnp
import haiku as hk
import pickle

from core.dynamics.carla_4state import CarlaDynamics

PATH = './results-less-conservative/trained_cbf.npy'
DELTA_F = 0.3
DELTA_G = 0.4

CTE_MAX = 1.6175946161054822
SPEED_MAX = 7.285775632710312
THETA_E_MAX = 2.9999982774423595
D_MAX = 26.73716521658956
DTHETA_T_MAX = 0.8976939936192778
INPUT_MAX = 0.23236677428018998

def main():

    net = hk.without_apply_rng(hk.transform(lambda x: net_fn()(x)))
    with open(PATH, 'rb') as handle:
        loaded_params = pickle.load(handle)

    learned_h = lambda x: jnp.sum(net.apply(loaded_params, x))
    zero_ctrl = get_zero_controller()

    safe_ctrl = make_safe_controller(zero_ctrl, learned_h)

    # Example usage:
    u = safe_ctrl(jnp.array([1., 2., 3., 4.]), 1)

def make_safe_controller(nominal_ctrl, h):
    """Create a safe controller using learned hybrid CBF."""

    dh = jax.grad(h, argnums=0)
    dyn = CarlaDynamics()
    alpha = lambda x : x
    norm = lambda x : jnp.linalg.norm(x)
    cpnorm = lambda x : cp.norm(x)
    dot = lambda x, y : jnp.dot(x, y)

    def safe_ctrl(x, d):
        """Solves HCBF-QP to map an input state to a safe action u.
        
        Params:
            x: state.
            d: disturbance.
        """

        cte, v, θ_e, d_var = x
        x = jnp.array([
            cte / CTE_MAX, 
            v / SPEED_MAX, 
            θ_e / THETA_E_MAX, 
            d_var / D_MAX
        ]).reshape(x.shape)

        d /= DTHETA_T_MAX

        # compute action used by nominal controller
        u_nom = nominal_ctrl(x)

        # setup and solve HCBF-QP with CVXPY
        u_mod = cp.Variable(len(u_nom))
        obj = cp.Minimize(cp.sum_squares(u_mod - u_nom))
        constraints = [
            dot(dh(x), dyn.f(x, d)) + u_mod.T @ dot(dyn.g(x).T, dh(x)) + alpha(h(x)) - norm(dh(x)) * (DELTA_F + DELTA_G * cpnorm(u_mod)) >= 0
        ]
        
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, verbose=True, max_iters=20000, eps=1e-10)

        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return u_mod.value
        return jnp.array([0.])

    return safe_ctrl

def net_fn(net_dims=[32, 16]):
    """Feed-forward NN architecture."""

    layers = []
    for dim in net_dims:
        layers.extend([hk.Linear(dim), jnp.tanh])
    layers.append(hk.Linear(1))

    return hk.Sequential(layers)

def get_zero_controller():
    """Returns a zero controller"""

    return lambda state: jnp.array([0.])

if __name__ == '__main__':
    main()