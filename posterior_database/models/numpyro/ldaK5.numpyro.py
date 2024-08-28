from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import log_real, log_sum_exp_array

def convert_inputs(inputs):
    N = inputs['N']
    V = inputs['V']
    w = array(inputs['w'], dtype=dtype_long)
    M = inputs['M']
    doc = array(inputs['doc'], dtype=dtype_long)
    alpha = array(inputs['alpha'], dtype=dtype_float)
    beta__ = array(inputs['beta'], dtype=dtype_float)
    return { 'N': N, 'V': V, 'w': w, 'M': M, 'doc': doc, 'alpha': alpha,
             'beta__': beta__ }

def model(*, N, V, w, M, doc, alpha, beta__):
    # Parameters
    theta = sample('theta', simplex_constrained_improper_uniform(shape=[
    M, 5]))
    phi = sample('phi', simplex_constrained_improper_uniform(shape=[5, V]))
    # Model
    def _fori__1(m, _acc__2):
        observe(f'_theta__{m}__3', dirichlet(alpha), theta[m - 1])
        return None
    _ = fori_loop(1, M + 1, _fori__1, None)
    def _fori__4(k, _acc__5):
        observe(f'_phi__{k}__6', dirichlet(beta__), phi[k - 1])
        return None
    _ = fori_loop(1, 5 + 1, _fori__4, None)
    def _fori__7(n, _acc__8):
        gamma__ = empty([5], dtype=dtype_float)
        @jit
        def _fori__9(k, _acc__10):
            gamma__ = _acc__10
            gamma__ = ops_index_update(gamma__, ops_index[k - 1], log_real(
            theta[doc[n - 1] - 1, k - 1]) + log_real(phi[k - 1, w[n - 1] - 1]))
            return gamma__
        gamma__ = lax_fori_loop(1, 5 + 1, _fori__9, gamma__)
        factor(f'_expr__{n}__11', log_sum_exp_array(gamma__))
        return None
    _ = fori_loop(1, N + 1, _fori__7, None)

def parameters_info(*, N, V, w, M, doc, alpha, beta__):
    return { 'theta': { 'shape': [M, 5] },'phi': { 'shape': [5, V] }, }

