from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import sqrt_real


def convert_inputs(inputs):
    I = inputs['I']
    n = array(inputs['n'], dtype=dtype_long)
    N = array(inputs['N'], dtype=dtype_long)
    x1 = array(inputs['x1'], dtype=dtype_float)
    x2 = array(inputs['x2'], dtype=dtype_float)
    return {'I': I, 'n': n, 'N': N, 'x1': x1, 'x2': x2}


def transformed_data(*, I, n, N, x1, x2):
    x1x2 = x1 * x2
    return {'x1x2': x1x2}


def model(*, I, n, N, x1, x2, x1x2):
    alpha0 = sample('alpha0', improper_uniform(shape=[]))
    alpha1 = sample('alpha1', improper_uniform(shape=[]))
    alpha12 = sample('alpha12', improper_uniform(shape=[]))
    alpha2 = sample('alpha2', improper_uniform(shape=[]))
    tau = sample('tau', lower_constrained_improper_uniform(0, shape=[]))
    b = sample('b', improper_uniform(shape=[I]))
    sigma = true_divide(array(1.0, dtype=dtype_float), sqrt_real(tau))
    observe('_alpha0__1', normal(array(0.0, dtype=dtype_float), array(
        1000.0, dtype=dtype_float)), alpha0)
    observe('_alpha1__2', normal(array(0.0, dtype=dtype_float), array(
        1000.0, dtype=dtype_float)), alpha1)
    observe('_alpha2__3', normal(array(0.0, dtype=dtype_float), array(
        1000.0, dtype=dtype_float)), alpha2)
    observe('_alpha12__4', normal(array(0.0, dtype=dtype_float), array(
        1000.0, dtype=dtype_float)), alpha12)
    observe('_tau__5', gamma(array(0.001, dtype=dtype_float), array(0.001,
        dtype=dtype_float)), tau)
    observe('_b__6', normal(array(0.0, dtype=dtype_float), sigma), b)
    observe('_n__7', binomial_logit(N, alpha0 + alpha1 * x1 + alpha2 * x2 +
        alpha12 * x1x2 + b), n)


def generated_quantities(*, I, n, N, x1, x2, x1x2, alpha0, alpha1, alpha12,
    alpha2, tau, b):
    sigma = true_divide(array(1.0, dtype=dtype_float), sqrt_real(tau))
    return {'sigma': sigma}


def map_generated_quantities(_samples, *, I, n, N, x1, x2, x1x2):

    def _generated_quantities(alpha0, alpha1, alpha12, alpha2, tau, b):
        return generated_quantities(I=I, n=n, N=N, x1=x1, x2=x2, x1x2=x1x2,
            alpha0=alpha0, alpha1=alpha1, alpha12=alpha12, alpha2=alpha2,
            tau=tau, b=b)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['alpha0'], _samples['alpha1'], _samples['alpha12'],
        _samples['alpha2'], _samples['tau'], _samples['b'])


def parameters_info(*, I, n, N, x1, x2, x1x2):
    return {'alpha0': {'shape': []}, 'alpha1': {'shape': []}, 'alpha12': {
        'shape': []}, 'alpha2': {'shape': []}, 'tau': {'shape': []}, 'b': {
        'shape': [I]}}
