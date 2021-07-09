import cvxpy as cp
import jax
import jax.numpy as jnp
import haiku as hk
import pickle
import json
import os

from core.dynamics.carla_4state import CarlaDynamics

ROOT = 'results-left-turn-normalized'
CBF_PATH = os.path.join(ROOT, 'trained_cbf.npy')
ARGS_PATH = os.path.join(ROOT, 'args.json')
META_DATA_PATH = os.path.join(ROOT, 'meta_data.json')

def main():

    args_dict = load_json(ARGS_PATH)
    meta_data = load_json(META_DATA_PATH)

    net = hk.without_apply_rng(hk.transform(lambda x: net_fn()(x)))
    with open(CBF_PATH, 'rb') as handle:
        loaded_params = pickle.load(handle)

    learned_h = lambda x: jnp.sum(net.apply(loaded_params, x))
    zero_ctrl = get_zero_controller()

    safe_ctrl = make_safe_controller(zero_ctrl, learned_h, args_dict, meta_data)

    # Example usage:
    u = safe_ctrl(jnp.array([1., 2., 3., 4.]), 1)

def make_safe_controller(nominal_ctrl, h, args_dict, meta_data):
    """Create a safe controller using learned hybrid CBF."""

    delta_f, delta_g = args_dict['delta_f'], args_dict['delta_g']
    maxes = meta_data['normalizers']
    # T_x = jnp.diag(jnp.array([
    #     maxes['cte'], maxes['speed'], maxes['theta_e'], maxes['d']
    # ]))
    T_x = jnp.eye(4)

    dh = jax.grad(h, argnums=0)
    dyn = CarlaDynamics(T_x)
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
        # x = jnp.array([
        #     cte / float(maxes['cte']), 
        #     v / float(maxes['speed']), 
        #     θ_e / float(maxes['theta_e']), 
        #     d_var / float(maxes['d'])
        # ]).reshape(4,)

        # compute action used by nominal controller
        u_nom = nominal_ctrl(x)

        # setup and solve HCBF-QP with CVXPY
        u_mod = cp.Variable(len(u_nom))
        obj = cp.Minimize(cp.sum_squares(u_mod - u_nom))
        constraints = [
            dot(dh(x), dyn.f(x, d)) + u_mod.T @ dot(dyn.g(x).T, dh(x)) + alpha(h(x)) - norm(dh(x)) * (delta_f + delta_g * cpnorm(u_mod)) >= 0
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

def load_json(fname):
    with open(fname) as json_file:
        data = json.load(json_file)
    return data

if __name__ == '__main__':
    main()