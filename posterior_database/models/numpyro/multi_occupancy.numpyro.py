from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import exp_real, log1m_inv_logit_real, log_inv_logit_real, log_sum_exp_real_real, rep_vector_int_int, square_real

def cov_matrix_2d(sigma, rho):
    Sigma = empty([2, 2], dtype=dtype_float)
    Sigma = ops_index_update(Sigma, ops_index[1 - 1, 1 - 1], square_real(
    sigma[1 - 1]))
    Sigma = ops_index_update(Sigma, ops_index[2 - 1, 2 - 1], square_real(
    sigma[2 - 1]))
    Sigma = ops_index_update(Sigma, ops_index[1 - 1, 2 - 1], sigma[1 - 1] * sigma[
    2 - 1] * rho)
    Sigma = ops_index_update(Sigma, ops_index[2 - 1, 1 - 1], Sigma[1 - 1,
                                                                   2 - 1])
    return Sigma

def lp_observed(X, K, logit_psi, logit_theta):
    return log_inv_logit_real(logit_psi) + binomial_logit_lpmf(X, K,
                                                               logit_theta)

def lp_unobserved(K, logit_psi, logit_theta):
    return log_sum_exp_real_real(lp_observed(0, K, logit_psi, logit_theta),
                                 log1m_inv_logit_real(logit_psi))

def lp_never_observed(J, K, logit_psi, logit_theta, Omega):
    lp_unavailable = bernoulli_lpmf(0, Omega)
    lp_available = bernoulli_lpmf(1, Omega) + J * lp_unobserved(K, logit_psi,
                                                                logit_theta)
    return log_sum_exp_real_real(lp_unavailable, lp_available)

def convert_inputs(inputs):
    J = inputs['J']
    K = inputs['K']
    n = inputs['n']
    X = array(inputs['X'], dtype=dtype_long)
    S = inputs['S']
    return { 'J': J, 'K': K, 'n': n, 'X': X, 'S': S }

def model(*, J, K, n, X, S):
    # Parameters
    alpha = sample('alpha', improper_uniform(shape=[]))
    beta__ = sample('beta', improper_uniform(shape=[]))
    Omega = sample('Omega', uniform(0, 1))
    rho_uv = sample('rho_uv', uniform(- 1, 1))
    sigma_uv = sample('sigma_uv', lower_constrained_improper_uniform(0, shape=[
    2]))
    uv1 = sample('uv1', improper_uniform(shape=[S]))
    uv2 = sample('uv2', improper_uniform(shape=[S]))
    # Transformed parameters
    uv = empty([S, 2], dtype=dtype_float)
    logit_psi = empty([S], dtype=dtype_float)
    @jit
    def _fori__1(i, _acc__2):
        (logit_psi, uv) = _acc__2
        uv = ops_index_update(uv, ops_index[i - 1, 1 - 1], uv1[i - 1])
        uv = ops_index_update(uv, ops_index[i - 1, 2 - 1], uv2[i - 1])
        logit_psi = ops_index_update(logit_psi, ops_index[i - 1], uv[
        i - 1, 1 - 1] + alpha)
        return (logit_psi, uv)
    (logit_psi, uv) = lax_fori_loop(1, S + 1, _fori__1, (logit_psi, uv))
    logit_theta = empty([S], dtype=dtype_float)
    @jit
    def _fori__3(i, _acc__4):
        logit_theta = _acc__4
        logit_theta = ops_index_update(logit_theta, ops_index[i - 1], uv[
        i - 1, 2 - 1] + beta__)
        return logit_theta
    logit_theta = lax_fori_loop(1, S + 1, _fori__3, logit_theta)
    # Model
    observe('_alpha__5', cauchy(0, array(2.5, dtype=dtype_float)), alpha)
    observe('_beta__6', cauchy(0, array(2.5, dtype=dtype_float)), beta__)
    observe('_sigma_uv__7', cauchy(0, array(2.5, dtype=dtype_float)), sigma_uv)
    observe('_expr__8', beta(2, 2), true_divide((rho_uv + 1), 2))
    factor('_expr__9', multi_normal_lpdf(uv, rep_vector_int_int(0, 2),
                                         cov_matrix_2d(sigma_uv, rho_uv)))
    observe('_Omega__10', beta(2, 2), Omega)
    def _fori__11(i, _acc__12):
        observe(f'_1__{i}__13', bernoulli(Omega), 1)
        def _fori__14(j, _acc__15):
            def _then__16(_acc__17):
                factor(f'_expr__{j}__{i}__20', lp_observed(X[i - 1, j - 1],
                                                           K,
                                                           logit_psi[
                                                           i - 1],
                                                           logit_theta[
                                                           i - 1]))
                return None
            def _else__18(_acc__19):
                factor(f'_expr__{j}__{i}__21', lp_unobserved(K,
                                                             logit_psi[
                                                             i - 1],
                                                             logit_theta[
                                                             i - 1]))
                return None
            _ = numpyro_cond(X[i - 1, j - 1] > 0, _then__16, _else__18, None)
            return None
        _ = fori_loop(1, J + 1, _fori__14, None)
        return None
    _ = fori_loop(1, n + 1, _fori__11, None)
    def _fori__22(i, _acc__23):
        factor(f'_expr__{i}__24', lp_never_observed(J, K, logit_psi[i - 1],
                                                    logit_theta[i - 1], Omega))
        return None
    _ = fori_loop((n + 1), S + 1, _fori__22, None)


def generated_quantities(*, J, K, n, X, S, alpha, beta__, Omega, rho_uv,
                            sigma_uv, uv1, uv2):
    # Transformed parameters
    uv = empty([S, 2], dtype=dtype_float)
    logit_psi = empty([S], dtype=dtype_float)
    @jit
    def _fori__25(i, _acc__26):
        (logit_psi, uv) = _acc__26
        uv = ops_index_update(uv, ops_index[i - 1, 1 - 1], uv1[i - 1])
        uv = ops_index_update(uv, ops_index[i - 1, 2 - 1], uv2[i - 1])
        logit_psi = ops_index_update(logit_psi, ops_index[i - 1], uv[
        i - 1, 1 - 1] + alpha)
        return (logit_psi, uv)
    (logit_psi, uv) = lax_fori_loop(1, S + 1, _fori__25, (logit_psi, uv))
    logit_theta = empty([S], dtype=dtype_float)
    @jit
    def _fori__27(i, _acc__28):
        logit_theta = _acc__28
        logit_theta = ops_index_update(logit_theta, ops_index[i - 1], uv[
        i - 1, 2 - 1] + beta__)
        return logit_theta
    logit_theta = lax_fori_loop(1, S + 1, _fori__27, logit_theta)
    # Generated quantities
    E_N = S * Omega
    E_N_2 = n
    @jit
    def _fori__29(i, _acc__30):
        E_N_2 = _acc__30
        lp_unavailable = bernoulli_lpmf(0, Omega)
        lp_available = bernoulli_lpmf(1, Omega) + J * lp_unobserved(K,
                                                                    logit_psi[
                                                                    i - 1],
                                                                    logit_theta[
                                                                    i - 1])
        Pr_available = exp_real(lp_available - log_sum_exp_real_real(
                                lp_unavailable, lp_available))
        E_N_2 = E_N_2 + bernoulli_rng(Pr_available)
        return E_N_2
    E_N_2 = lax_fori_loop((n + 1), S + 1, _fori__29, E_N_2)
    sim_uv = multi_normal_rng(rep_vector_int_int(0, 2),
                              cov_matrix_2d(sigma_uv, rho_uv))
    logit_psi_sim = alpha + sim_uv[1 - 1]
    logit_theta_sim = beta__ + sim_uv[2 - 1]
    return { 'uv': uv, 'logit_psi': logit_psi, 'logit_theta': logit_theta,
             'E_N': E_N, 'E_N_2': E_N_2, 'sim_uv': sim_uv,
             'logit_psi_sim': logit_psi_sim,
             'logit_theta_sim': logit_theta_sim }

def map_generated_quantities(_samples, *, J, K, n, X, S):
    def _generated_quantities(alpha, beta__, Omega, rho_uv, sigma_uv, uv1,
                              uv2):
        return generated_quantities(J=J, K=K, n=n, X=X, S=S, alpha=alpha,
                                    beta__=beta__, Omega=Omega,
                                    rho_uv=rho_uv, sigma_uv=sigma_uv,
                                    uv1=uv1, uv2=uv2)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['alpha'], _samples['beta'], _samples['Omega'],
              _samples['rho_uv'], _samples['sigma_uv'], _samples['uv1'],
              _samples['uv2'])

def parameters_info(*, J, K, n, X, S):
    return { 'alpha': { 'shape': [] },'beta': { 'shape': [] },
             'Omega': { 'shape': [] },'rho_uv': { 'shape': [] },
             'sigma_uv': { 'shape': [2] },'uv1': { 'shape': [S] },
             'uv2': { 'shape': [S] }, }

