import jax.numpy as jnp

class CarlaDynamics:
    def __init__(self):
        self._state_dim = 4
        self._input_dim = 1

    @property
    def state_dim(self):
        return self._state_dim
    
    @property
    def input_dim(self):
        return self._input_dim

    def f(self, state, disturbance):
        c, v, θ_e, d = state
        ϕ_dot_t = disturbance
        return jnp.array([
            v * jnp.sin(θ_e), 
            -1.0954 * v - 0.007 * v ** 2 - 0.1521 * d + 3.37387,
            -ϕ_dot_t,
            3.6 * v - 20
        ]).reshape(self._state_dim, 1)

    def g(self, state):
        c, v, θ, d = state
        return jnp.array([
            0., 0., v / 2.51, 0.
        ]).reshape(self._state_dim, 1)