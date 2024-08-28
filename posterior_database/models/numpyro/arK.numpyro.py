from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element

def convert_inputs(inputs):
    K = inputs['K']
    T = inputs['T']
    y = array(inputs['y'], dtype=dtype_float)
    return { 'K': K, 'T': T, 'y': y }

def model(*, K, T, y):
    # Parameters
    alpha = sample('alpha', improper_uniform(shape=[]))
    beta__ = sample('beta', improper_uniform(shape=[K]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    # Model
    observe('_alpha__1', normal(0, 10), alpha)
    observe('_beta__2', normal(0, 10), beta__)
    observe('_sigma__3', cauchy(0, array(2.5, dtype=dtype_float)), sigma)
    def _fori__4(t, _acc__5):
        mu = alpha
        @jit
        def _fori__6(k, _acc__7):
            mu = _acc__7
            mu = mu + beta__[k - 1] * y[t - k - 1]
            return mu
        mu = lax_fori_loop(1, K + 1, _fori__6, mu)
        observe(f'_y__{t}__8', normal(mu, sigma), y[t - 1])
        return None
    _ = fori_loop((K + 1), T + 1, _fori__4, None)

def parameters_info(*, K, T, y):
    return { 'alpha': { 'shape': [] },'beta': { 'shape': [K] },
             'sigma': { 'shape': [] }, }

