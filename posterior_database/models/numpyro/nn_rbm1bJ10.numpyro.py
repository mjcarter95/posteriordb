from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import append_col_vector_matrix, append_row_rowvector_matrix, rep_vector_int_int, sqrt_real, tanh_matrix, to_vector_matrix

def convert_inputs(inputs):
    M = inputs['M']
    N = inputs['N']
    x = array(inputs['x'], dtype=dtype_float)
    K = inputs['K']
    y = array(inputs['y'], dtype=dtype_long)
    return { 'M': M, 'N': N, 'x': x, 'K': K, 'y': y }

def transformed_data(*, M, N, x, K, y):
    # Transformed data
    J = 10
    nu_alpha = array(0.5, dtype=dtype_float)
    s2_0_alpha = (true_divide(array(0.05, dtype=dtype_float), M ** (true_divide(1, nu_alpha)))) ** 2
    nu_beta = array(0.5, dtype=dtype_float)
    s2_0_beta = (true_divide(array(0.05, dtype=dtype_float), J ** (true_divide(1, nu_beta)))) ** 2
    ones__ = rep_vector_int_int(1, N)
    x1 = append_col_vector_matrix(ones__, x)
    return { 'J': J, 'nu_alpha': nu_alpha, 's2_0_alpha': s2_0_alpha,
             'nu_beta': nu_beta, 's2_0_beta': s2_0_beta, 'ones__': ones__,
             'x1': x1 }

def model(*, M, N, x, K, y, J, nu_alpha, s2_0_alpha, nu_beta, s2_0_beta,
             ones__, x1):
    # Parameters
    sigma2_alpha = sample('sigma2_alpha', lower_constrained_improper_uniform(0, shape=[]))
    sigma2_beta = sample('sigma2_beta', lower_constrained_improper_uniform(0, shape=[]))
    alpha = sample('alpha', improper_uniform(shape=[M, J]))
    beta__ = sample('beta', improper_uniform(shape=[J, K - 1]))
    alpha1 = sample('alpha1', improper_uniform(shape=[J]))
    beta1 = sample('beta1', improper_uniform(shape=[K - 1]))
    # Model
    v = append_col_vector_matrix(ones__,
                                 matmul(append_col_vector_matrix(ones__,
                                                                 tanh_matrix(
                                                                 matmul(x1, append_row_rowvector_matrix(
                                                                 alpha1,
                                                                 alpha)))), append_row_rowvector_matrix(
                                 beta1, beta__)))
    observe('_alpha1__1', normal(0, 1), alpha1)
    observe('_beta1__2', normal(0, 1), beta1)
    observe('_sigma2_alpha__3', inv_gamma(true_divide(nu_alpha, 2),
                                          true_divide(nu_alpha * s2_0_alpha, 2)), sigma2_alpha)
    observe('_sigma2_beta__4', inv_gamma(true_divide(nu_beta, 2),
                                         true_divide(nu_beta * s2_0_beta, 2)), sigma2_beta)
    observe('_expr__5', normal(0, sqrt_real(sigma2_alpha)), to_vector_matrix(
    alpha))
    observe('_expr__6', normal(0, sqrt_real(sigma2_beta)), to_vector_matrix(
    beta__))
    def _fori__7(n, _acc__8):
        observe(f'_y__{n}__9', categorical_logit(v[n - 1]), y[n - 1] - 1)
        return None
    _ = fori_loop(1, N + 1, _fori__7, None)

def parameters_info(*, M, N, x, K, y, J, nu_alpha, s2_0_alpha, nu_beta,
                       s2_0_beta, ones__, x1):
    return { 'sigma2_alpha': { 'shape': [] },'sigma2_beta': { 'shape': [] },
             'alpha': { 'shape': [M, J] },'beta': { 'shape': [J, K - 1] },
             'alpha1': { 'shape': [J] },'beta1': { 'shape': [K - 1] }, }

