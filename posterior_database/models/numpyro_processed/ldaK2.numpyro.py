from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import log_real, log_sum_exp_array


def convert_inputs(inputs):
    V = inputs['V']
    N = inputs['N']
    w = array(inputs['w'], dtype=dtype_long)
    M = inputs['M']
    doc = array(inputs['doc'], dtype=dtype_long)
    return {'V': V, 'N': N, 'w': w, 'M': M, 'doc': doc}


def transformed_data(*, V, N, w, M, doc):
    K = 2
    alpha = empty([K], dtype=dtype_float)

    @jit
    def _fori__1(k, _acc__2):
        alpha = _acc__2
        alpha = alpha.at[k - 1].set(1)
        return alpha
    alpha = lax_fori_loop(1, K + 1, _fori__1, alpha)
    beta__ = empty([V], dtype=dtype_float)

    @jit
    def _fori__3(v, _acc__4):
        beta__ = _acc__4
        beta__ = beta__.at[v - 1].set(1)
        return beta__
    beta__ = lax_fori_loop(1, V + 1, _fori__3, beta__)
    return {'K': K, 'alpha': alpha, 'beta__': beta__}


def model(*, V, N, w, M, doc, K, alpha, beta__):
    theta = sample('theta', simplex_constrained_improper_uniform(shape=[M, K]))
    phi = sample('phi', simplex_constrained_improper_uniform(shape=[K, V]))

    def _fori__5(m, _acc__6):
        observe(f'_theta__{m}__7', dirichlet(alpha), theta[m - 1])
        return None
    _ = fori_loop(1, M + 1, _fori__5, None)

    def _fori__8(k, _acc__9):
        observe(f'_phi__{k}__10', dirichlet(beta__), phi[k - 1])
        return None
    _ = fori_loop(1, K + 1, _fori__8, None)

    def _fori__11(n, _acc__12):
        gamma__ = empty([K], dtype=dtype_float)

        @jit
        def _fori__13(k, _acc__14):
            gamma__ = _acc__14
            gamma__ = gamma__.at[k - 1].set(log_real(theta[doc[n - 1] - 1, 
                k - 1]) + log_real(phi[k - 1, w[n - 1] - 1]))
            return gamma__
        gamma__ = lax_fori_loop(1, K + 1, _fori__13, gamma__)
        factor(f'_expr__{n}__15', log_sum_exp_array(gamma__))
        return None
    _ = fori_loop(1, N + 1, _fori__11, None)


def parameters_info(*, V, N, w, M, doc, K, alpha, beta__):
    return {'theta': {'shape': [M, K]}, 'phi': {'shape': [K, V]}}
