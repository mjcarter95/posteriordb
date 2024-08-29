from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import cholesky_decompose_matrix, diag_matrix_vector, rep_vector_real_int


def convert_inputs(inputs):
    N = inputs['N']
    x = array(inputs['x'], dtype=dtype_float)
    k = array(inputs['k'], dtype=dtype_long)
    return {'N': N, 'x': x, 'k': k}


def model(*, N, x, k):
    rho = sample('rho', lower_constrained_improper_uniform(0, shape=[]))
    alpha = sample('alpha', lower_constrained_improper_uniform(0, shape=[]))
    f_tilde = sample('f_tilde', improper_uniform(shape=[N]))
    cov = gp_exp_quad_cov_array_real_real(x, alpha, rho) + diag_matrix_vector(
        rep_vector_real_int(array(1e-10, dtype=dtype_float), N))
    L_cov = cholesky_decompose_matrix(cov)
    f = matmul(L_cov, f_tilde)
    observe('_rho__1', gamma(25, 4), rho)
    observe('_alpha__2', normal(0, 2), alpha)
    observe('_f_tilde__3', normal(0, 1), f_tilde)
    observe('_k__4', poisson_log(f), k)


def generated_quantities(*, N, x, k, rho, alpha, f_tilde):
    cov = gp_exp_quad_cov_array_real_real(x, alpha, rho) + diag_matrix_vector(
        rep_vector_real_int(array(1e-10, dtype=dtype_float), N))
    L_cov = cholesky_decompose_matrix(cov)
    f = matmul(L_cov, f_tilde)
    return {'cov': cov, 'L_cov': L_cov, 'f': f}


def map_generated_quantities(_samples, *, N, x, k):

    def _generated_quantities(rho, alpha, f_tilde):
        return generated_quantities(N=N, x=x, k=k, rho=rho, alpha=alpha,
            f_tilde=f_tilde)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['rho'], _samples['alpha'], _samples['f_tilde'])


def parameters_info(*, N, x, k):
    return {'rho': {'shape': []}, 'alpha': {'shape': []}, 'f_tilde': {
        'shape': [N]}}
