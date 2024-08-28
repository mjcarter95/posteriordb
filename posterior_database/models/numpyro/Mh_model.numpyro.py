from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import inv_logit_real, log_sum_exp_real_real, logit_real, sum_array

def convert_inputs(inputs):
    M = inputs['M']
    T = inputs['T']
    y = array(inputs['y'], dtype=dtype_long)
    return { 'M': M, 'T': T, 'y': y }

def transformed_data(*, M, T, y):
    # Transformed data
    C = 0
    @jit
    def _fori__1(i, _acc__2):
        C = _acc__2
        @jit
        def _then__3(_acc__4):
            C = _acc__4
            C = C + 1
            return C
        @jit
        def _else__5(acc):
            return acc
        C = lax_cond(y[i - 1] > 0, _then__3, _else__5, C)
        return C
    C = lax_fori_loop(1, M + 1, _fori__1, C)
    return { 'C': C }

def model(*, M, T, y, C):
    # Parameters
    omega = sample('omega', uniform(0, 1))
    mean_p = sample('mean_p', uniform(0, 1))
    sigma = sample('sigma', uniform(0, 5))
    eps_raw = sample('eps_raw', improper_uniform(shape=[M]))
    # Transformed parameters
    eps = logit_real(mean_p) + sigma * eps_raw
    # Model
    observe('_eps_raw__6', normal(0, 1), eps_raw)
    def _fori__7(i, _acc__8):
        def _then__9(_acc__10):
            factor(f'_expr__{i}__13', bernoulli_lpmf(1, omega) + binomial_logit_lpmf(
            y[i - 1], T, eps[i - 1]))
            return None
        def _else__11(_acc__12):
            factor(f'_expr__{i}__14', log_sum_exp_real_real(bernoulli_lpmf(
                                                            1, omega) + binomial_logit_lpmf(
                                                            0, T, eps[
                                                            i - 1]),
                                                            bernoulli_lpmf(
                                                            0, omega)))
            return None
        _ = numpyro_cond(y[i - 1] > 0, _then__9, _else__11, None)
        return None
    _ = fori_loop(1, M + 1, _fori__7, None)


def generated_quantities(*, M, T, y, C, omega, mean_p, sigma, eps_raw):
    # Transformed parameters
    eps = logit_real(mean_p) + sigma * eps_raw
    # Generated quantities
    z = empty([M], dtype=dtype_long)
    @jit
    def _fori__15(i, _acc__16):
        z = _acc__16
        @jit
        def _then__17(_acc__18):
            z = _acc__18
            z = ops_index_update(z, ops_index[i - 1], 1)
            return z
        @jit
        def _else__19(_acc__20):
            z = _acc__20
            qT = inv_logit_real(- eps[i - 1]) ** T
            z = ops_index_update(z, ops_index[i - 1], bernoulli_rng(true_divide(omega * qT, (omega * qT + (1 - omega)))))
            return z
        z = lax_cond(y[i - 1] > 0, _then__17, _else__19, z)
        return z
    z = lax_fori_loop(1, M + 1, _fori__15, z)
    N = sum_array(z)
    return { 'eps': eps, 'z': z, 'N': N }

def map_generated_quantities(_samples, *, M, T, y, C):
    def _generated_quantities(omega, mean_p, sigma, eps_raw):
        return generated_quantities(M=M, T=T, y=y, C=C, omega=omega,
                                    mean_p=mean_p, sigma=sigma,
                                    eps_raw=eps_raw)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['omega'], _samples['mean_p'], _samples['sigma'],
              _samples['eps_raw'])

def parameters_info(*, M, T, y, C):
    return { 'omega': { 'shape': [] },'mean_p': { 'shape': [] },
             'sigma': { 'shape': [] },'eps_raw': { 'shape': [M] }, }

