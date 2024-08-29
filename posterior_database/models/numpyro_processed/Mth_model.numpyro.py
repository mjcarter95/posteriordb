from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import inv_logit_matrix, log_sum_exp_real_real, logit_array, prod_vector, rep_vector_int_int, sum_array


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
    sigma = sample('sigma', uniform(0, 5))
    eps_raw = sample('eps_raw', improper_uniform(shape=[M]))
    eps = sigma * eps_raw
    mean_lp = logit_array(mean_p)
    logit_p = empty([M, T], dtype=dtype_float)

    @jit
    def _fori__6(j, _acc__7):
        logit_p = _acc__7
        logit_p = logit_p.at[:, j - 1].set(mean_lp[j - 1] + eps)
        return logit_p
    logit_p = lax_fori_loop(1, T + 1, _fori__6, logit_p)
    observe('_eps_raw__8', normal(0, 1), eps_raw)

    def _fori__9(i, _acc__10):

        def _then__11(_acc__12):
            factor(f'_expr__{i}__15', bernoulli_lpmf(1, omega) +
                bernoulli_logit_lpmf(y[i - 1], logit_p[i - 1]))
            return None

        def _else__13(_acc__14):
            factor(f'_expr__{i}__16', log_sum_exp_real_real(bernoulli_lpmf(
                1, omega) + bernoulli_logit_lpmf(y[i - 1], logit_p[i - 1]),
                bernoulli_lpmf(0, omega)))
            return None
        _ = numpyro_cond(s[i - 1] > 0, _then__11, _else__13, None)
        return None
    _ = fori_loop(1, M + 1, _fori__9, None)


def generated_quantities(*, M, T, y, C, s, omega, mean_p, sigma, eps_raw):
    eps = sigma * eps_raw
    mean_lp = logit_array(mean_p)
    logit_p = empty([M, T], dtype=dtype_float)

    @jit
    def _fori__17(j, _acc__18):
        logit_p = _acc__18
        logit_p = logit_p.at[:, j - 1].set(mean_lp[j - 1] + eps)
        return logit_p
    logit_p = lax_fori_loop(1, T + 1, _fori__17, logit_p)
    p = inv_logit_matrix(logit_p)
    z = empty([M], dtype=dtype_long)

    @jit
    def _fori__19(i, _acc__20):
        z = _acc__20

        @jit
        def _then__21(_acc__22):
            z = _acc__22
            z = z.at[i - 1].set(1)
            return z

        @jit
        def _else__23(_acc__24):
            z = _acc__24
            pr = prod_vector(rep_vector_int_int(1, T) - p[i - 1])
            z = z.at[i - 1].set(bernoulli_rng(true_divide(omega * pr, omega *
                pr + (1 - omega))))
            return z
        z = lax_cond(s[i - 1] > 0, _then__21, _else__23, z)
        return z
    z = lax_fori_loop(1, M + 1, _fori__19, z)
    N = sum_array(z)
    return {'eps': eps, 'mean_lp': mean_lp, 'logit_p': logit_p, 'p': p, 'z':
        z, 'N': N}


def map_generated_quantities(_samples, *, M, T, y, C, s):

    def _generated_quantities(omega, mean_p, sigma, eps_raw):
        return generated_quantities(M=M, T=T, y=y, C=C, s=s, omega=omega,
            mean_p=mean_p, sigma=sigma, eps_raw=eps_raw)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['omega'], _samples['mean_p'], _samples['sigma'],
        _samples['eps_raw'])


def parameters_info(*, M, T, y, C, s):
    return {'omega': {'shape': []}, 'mean_p': {'shape': [T]}, 'sigma': {
        'shape': []}, 'eps_raw': {'shape': [M]}}
