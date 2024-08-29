from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import exp_vector, square_vector


def convert_inputs(inputs):
    n = inputs['n']
    C = array(inputs['C'], dtype=dtype_long)
    year = array(inputs['year'], dtype=dtype_float)
    return {'n': n, 'C': C, 'year': year}


def transformed_data(*, n, C, year):
    year_squared = square_vector(year)
    year_cubed = year_squared * year
    return {'year_squared': year_squared, 'year_cubed': year_cubed}


def model(*, n, C, year, year_squared, year_cubed):
    alpha = sample('alpha', uniform(-20, 20))
    beta1 = sample('beta1', uniform(-10, 10))
    beta2 = sample('beta2', uniform(-10, 10))
    beta3 = sample('beta3', uniform(-10, 10))
    log_lambda = (alpha + beta1 * year + +beta2 * year_squared + +beta3 *
        year_cubed)
    observe('_C__1', poisson_log(log_lambda), C)


def generated_quantities(*, n, C, year, year_squared, year_cubed, alpha,
    beta1, beta2, beta3):
    log_lambda = (alpha + beta1 * year + +beta2 * year_squared + +beta3 *
        year_cubed)
    lambda__ = exp_vector(log_lambda)
    return {'log_lambda': log_lambda, 'lambda': lambda__}


def map_generated_quantities(_samples, *, n, C, year, year_squared, year_cubed
    ):

    def _generated_quantities(alpha, beta1, beta2, beta3):
        return generated_quantities(n=n, C=C, year=year, year_squared=
            year_squared, year_cubed=year_cubed, alpha=alpha, beta1=beta1,
            beta2=beta2, beta3=beta3)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['alpha'], _samples['beta1'], _samples['beta2'],
        _samples['beta3'])


def parameters_info(*, n, C, year, year_squared, year_cubed):
    return {'alpha': {'shape': []}, 'beta1': {'shape': []}, 'beta2': {
        'shape': []}, 'beta3': {'shape': []}}
