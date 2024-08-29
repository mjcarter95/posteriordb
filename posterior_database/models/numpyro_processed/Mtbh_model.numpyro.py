from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import inv_logit_matrix, log_sum_exp_real_real, logit_vector, prod_vector, rep_vector_int_int, sum_array


def convert_inputs(inputs):
    M = inputs['M']
    T = inputs['T']
    y = array(inputs['y'], dtype=dtype_long)
    return {'M': M, 'T': T, 'y': y}


def transformed_data(*, M, T, y):
    C = 0
    s = empty([M], dtype=dtype_long)

    @jit
    def _fori__1(i, _acc__2):
        C, s = _acc__2
        s = s.at[i - 1].set(sum_array(y[i - 1]))

        @jit
        def _then__3(_acc__4):
            C = _acc__4
            C = C + 1
            return C

        @jit
        def _else__5(acc):
            return acc
        C = lax_cond(s[i - 1] > 0, _then__3, _else__5, C)
        return C, s
    C, s = lax_fori_loop(1, M + 1, _fori__1, (C, s))
    return {'C': C, 's': s}


def model(*, M, T, y, C, s):
    omega = sample('omega', uniform(0, 1))
    mean_p = sample('mean_p', uniform(0 * ones([T]), 1))
    gamma__ = sample('gamma', improper_uniform(shape=[]))
    sigma = sample('sigma', uniform(0, 3))
    eps_raw = sample('eps_raw', improper_uniform(shape=[M]))
    eps = sigma * eps_raw
    alpha = logit_vector(mean_p)
    logit_p = empty([M, T], dtype=dtype_float)
    logit_p = logit_p.at[:, 1 - 1].set(alpha[1 - 1] + eps)

    @jit
    def _fori__6(i, _acc__7):
        logit_p = _acc__7

        @jit
        def _fori__8(j, _acc__9):
            logit_p = _acc__9
            logit_p = logit_p.at[i - 1, j - 1].set(alpha[j - 1] + eps[i - 1
                ] + gamma__ * y[i - 1, j - 1 - 1])
            return logit_p
        logit_p = lax_fori_loop(2, T + 1, _fori__8, logit_p)
        return logit_p
    logit_p = lax_fori_loop(1, M + 1, _fori__6, logit_p)
    observe('_gamma__10', normal(0, 10), gamma__)
    observe('_eps_raw__11', normal(0, 1), eps_raw)

    def _fori__12(i, _acc__13):

        def _then__14(_acc__15):
            factor(f'_expr__{i}__18', bernoulli_lpmf(1, omega) +
                bernoulli_logit_lpmf(y[i - 1], logit_p[i - 1]))
            return None

        def _else__16(_acc__17):
            factor(f'_expr__{i}__19', log_sum_exp_real_real(bernoulli_lpmf(
                1, omega) + bernoulli_logit_lpmf(y[i - 1], logit_p[i - 1]),
                bernoulli_lpmf(0, omega)))
            return None
        _ = numpyro_cond(s[i - 1] > 0, _then__14, _else__16, None)
        return None
    _ = fori_loop(1, M + 1, _fori__12, None)


def generated_quantities(*, M, T, y, C, s, omega, mean_p, gamma__, sigma,
    eps_raw):
    eps = sigma * eps_raw
    alpha = logit_vector(mean_p)
    logit_p = empty([M, T], dtype=dtype_float)
    logit_p = logit_p.at[:, 1 - 1].set(alpha[1 - 1] + eps)

    @jit
    def _fori__20(i, _acc__21):
        logit_p = _acc__21

        @jit
        def _fori__22(j, _acc__23):
            logit_p = _acc__23
            logit_p = logit_p.at[i - 1, j - 1].set(alpha[j - 1] + eps[i - 1
                ] + gamma__ * y[i - 1, j - 1 - 1])
            return logit_p
        logit_p = lax_fori_loop(2, T + 1, _fori__22, logit_p)
        return logit_p
    logit_p = lax_fori_loop(1, M + 1, _fori__20, logit_p)
    p = inv_logit_matrix(logit_p)
    z = empty([M], dtype=dtype_long)

    @jit
    def _fori__24(i, _acc__25):
        z = _acc__25

        @jit
        def _then__26(_acc__27):
            z = _acc__27
            z = z.at[i - 1].set(1)
            return z

        @jit
        def _else__28(_acc__29):
            z = _acc__29
            pr = prod_vector(rep_vector_int_int(1, T) - p[i - 1])
            z = z.at[i - 1].set(bernoulli_rng(true_divide(omega * pr, omega *
                pr + (1 - omega))))
            return z
        z = lax_cond(s[i - 1] > 0, _then__26, _else__28, z)
        return z
    z = lax_fori_loop(1, M + 1, _fori__24, z)
    N = sum_array(z)
    return {'eps': eps, 'alpha': alpha, 'logit_p': logit_p, 'p': p, 'z': z,
        'N': N}


def map_generated_quantities(_samples, *, M, T, y, C, s):

    def _generated_quantities(omega, mean_p, gamma__, sigma, eps_raw):
        return generated_quantities(M=M, T=T, y=y, C=C, s=s, omega=omega,
            mean_p=mean_p, gamma__=gamma__, sigma=sigma, eps_raw=eps_raw)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['omega'], _samples['mean_p'], _samples['gamma'],
        _samples['sigma'], _samples['eps_raw'])


def parameters_info(*, M, T, y, C, s):
    return {'omega': {'shape': []}, 'mean_p': {'shape': [T]}, 'gamma': {
        'shape': []}, 'sigma': {'shape': []}, 'eps_raw': {'shape': [M]}}
