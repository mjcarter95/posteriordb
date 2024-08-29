from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import log_real, log_sum_exp_vector, max_array, negative_infinity, softmax_vector


def convert_inputs(inputs):
    nmax = inputs['nmax']
    m = inputs['m']
    k = array(inputs['k'], dtype=dtype_long)
    return {'nmax': nmax, 'm': m, 'k': k}


def transformed_data(*, nmax, m, k):
    nmin = max_array(k)
    return {'nmin': nmin}


def model(*, nmax, m, k, nmin):
    theta = sample('theta', uniform(0, 1))
    lp_parts = empty([nmax], dtype=dtype_float)

    @jit
    def _fori__1(n, _acc__2):
        lp_parts = _acc__2

        @jit
        def _then__3(_acc__4):
            lp_parts = _acc__4
            lp_parts = lp_parts.at[n - 1].set(log_real(true_divide(array(
                1.0, dtype=dtype_float), nmax)) + negative_infinity())
            return lp_parts

        @jit
        def _else__5(_acc__6):
            lp_parts = _acc__6
            lp_parts = lp_parts.at[n - 1].set(log_real(true_divide(array(
                1.0, dtype=dtype_float), nmax)) + binomial_lpmf(k, n, theta))
            return lp_parts
        lp_parts = lax_cond(n < nmin, _then__3, _else__5, lp_parts)
        return lp_parts
    lp_parts = lax_fori_loop(1, nmax + 1, _fori__1, lp_parts)
    factor('_expr__7', log_sum_exp_vector(lp_parts))


def generated_quantities(*, nmax, m, k, nmin, theta):
    lp_parts = empty([nmax], dtype=dtype_float)

    @jit
    def _fori__8(n, _acc__9):
        lp_parts = _acc__9

        @jit
        def _then__10(_acc__11):
            lp_parts = _acc__11
            lp_parts = lp_parts.at[n - 1].set(log_real(true_divide(array(
                1.0, dtype=dtype_float), nmax)) + negative_infinity())
            return lp_parts

        @jit
        def _else__12(_acc__13):
            lp_parts = _acc__13
            lp_parts = lp_parts.at[n - 1].set(log_real(true_divide(array(
                1.0, dtype=dtype_float), nmax)) + binomial_lpmf(k, n, theta))
            return lp_parts
        lp_parts = lax_cond(n < nmin, _then__10, _else__12, lp_parts)
        return lp_parts
    lp_parts = lax_fori_loop(1, nmax + 1, _fori__8, lp_parts)
    prob_n = softmax_vector(lp_parts)
    n = categorical_rng(prob_n)
    return {'lp_parts': lp_parts, 'prob_n': prob_n, 'n': n}


def map_generated_quantities(_samples, *, nmax, m, k, nmin):

    def _generated_quantities(theta):
        return generated_quantities(nmax=nmax, m=m, k=k, nmin=nmin, theta=theta
            )
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['theta'])


def parameters_info(*, nmax, m, k, nmin):
    return {'theta': {'shape': []}}
