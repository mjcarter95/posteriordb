from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element

def convert_inputs(inputs):
    N = inputs['N']
    floor_measure = array(inputs['floor_measure'], dtype=dtype_float)
    log_radon = array(inputs['log_radon'], dtype=dtype_float)
    return { 'N': N, 'floor_measure': floor_measure, 'log_radon': log_radon }

def model(*, N, floor_measure, log_radon):
    # Parameters
    alpha = sample('alpha', improper_uniform(shape=[]))
    beta__ = sample('beta', improper_uniform(shape=[]))
    sigma_y = sample('sigma_y', lower_constrained_improper_uniform(0, shape=[]))
    # Model
    observe('_sigma_y__1', normal(0, 1), sigma_y)
    observe('_alpha__2', normal(0, 10), alpha)
    observe('_beta__3', normal(0, 10), beta__)
    mu = alpha + beta__ * floor_measure
    def _fori__4(n, _acc__5):
        factor(f'_expr__{n}__6', normal_lpdf(log_radon[n - 1], mu[n - 1],
                                             sigma_y))
        return None
    _ = fori_loop(1, N + 1, _fori__4, None)

def parameters_info(*, N, floor_measure, log_radon):
    return { 'alpha': { 'shape': [] },'beta': { 'shape': [] },
             'sigma_y': { 'shape': [] }, }

