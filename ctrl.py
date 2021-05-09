import cvxpy as cp
import jax
import jax.numpy as jnp
import haiku as hk
import pickle

from core.dynamics.carla_4state import CarlaDynamics

PATH = './results/trained_cbf.npy'

def main():

    net = hk.without_apply_rng(hk.transform(lambda x: net_fn()(x)))
    with open(PATH, 'rb') as handle:
        loaded_params = pickle.load(handle)

    learned_h = lambda x: jnp.sum(net.apply(loaded_params, x))
    zero_ctrl = get_zero_controller()

    safe_ctrl = make_safe_controller(zero_ctrl, learned_h)

    # Example usage:
    # u = safe_ctrl(jnp.array([1., 2., 3., 4.]), 1)

def make_safe_controller(nominal_ctrl, h):
    """Create a safe controller using learned hybrid CBF."""

    dh = jax.grad(h, argnums=0)
    dyn = CarlaDynamics()

    def safe_ctrl(x, d):
        """Solves HCBF-QP to map an input state to a safe action u.
        
        Params:
            x: state.
            d: disturbance.
        """

        # compute action used by nominal controller
        u_nom = nominal_ctrl(x)

        # compute function values
        f_of_x, g_of_x = dyn.f(x, d), dyn.g(x)
        h_of_x = h(x)
        dh_of_x = dh(x)

        # setup and solve HCBF-QP with CVXPY
        u_mod = cp.Variable(len(u_nom))
        obj = cp.Minimize(cp.sum_squares(u_mod - u_nom))
        constraints = [jnp.dot(dh_of_x, f_of_x) + u_mod.T @ jnp.dot(g_of_x.T, dh_of_x) + h_of_x >= 0]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, verbose=False, max_iters=20000, eps=1e-10)

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