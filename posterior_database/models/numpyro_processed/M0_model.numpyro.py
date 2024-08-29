from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import log_sum_exp_real_real, sum_array


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
    p = sample('p', uniform(0, 1))

    def _fori__6(i, _acc__7):

        def _then__8(_acc__9):
            factor(f'_expr__{i}__12', bernoulli_lpmf(1, omega) +
                binomial_lpmf(s[i - 1], T, p))
            return None

        def _else__10(_acc__11):
            factor(f'_expr__{i}__13', log_sum_exp_real_real(bernoulli_lpmf(
                1, omega) + binomial_lpmf(0, T, p), bernoulli_lpmf(0, omega)))
            return None
        _ = numpyro_cond(s[i - 1] > 0, _then__8, _else__10, None)
        return None
    _ = fori_loop(1, M + 1, _fori__6, None)


def generated_quantities(*, M, T, y, C, s, omega, p):
    omega_nd = true_divide(omega * (1 - p) ** T, omega * (1 - p) ** T + (1 -
        omega))
    N = C + binomial_rng(M - C, omega_nd)
    return {'omega_nd': omega_nd, 'N': N}


def map_generated_quantities(_samples, *, M, T, y, C, s):

    def _generated_quantities(omega, p):
        return generated_quantities(M=M, T=T, y=y, C=C, s=s, omega=omega, p=p)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['omega'], _samples['p'])


def parameters_info(*, M, T, y, C, s):
    return {'omega': {'shape': []}, 'p': {'shape': []}}
