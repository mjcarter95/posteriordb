from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import exp_vector

def convert_inputs(inputs):
    n = inputs['n']
    C = array(inputs['C'], dtype=dtype_long)
    year = array(inputs['year'], dtype=dtype_float)
    return { 'n': n, 'C': C, 'year': year }

def transformed_data(*, n, C, year):
    # Transformed data
    year_squared = year * year
    year_cubed = year * year * year
    return { 'year_squared': year_squared, 'year_cubed': year_cubed }

def model(*, n, C, year, year_squared, year_cubed):
    # Parameters
    alpha = sample('alpha', uniform(- 20, 20))
    beta1 = sample('beta1', uniform(- 10, 10))
    beta2 = sample('beta2', uniform(- 10, 20))
    beta3 = sample('beta3', uniform(- 10, 10))
    eps = sample('eps', improper_uniform(shape=[n]))
    sigma = sample('sigma', uniform(0, 5))
    # Transformed parameters
    log_lambda = alpha + beta1 * year + beta2 * year_squared + beta3 * year_cubed + eps
    # Model
    observe('_alpha__1', uniform(- 20, 20), alpha)
    observe('_beta1__2', uniform(- 10, 10), beta1)
    observe('_beta2__3', uniform(- 10, 10), beta2)
    observe('_beta3__4', uniform(- 10, 10), beta3)
    observe('_sigma__5', uniform(0, 5), sigma)
    observe('_C__6', poisson_log(log_lambda), C)
    observe('_eps__7', normal(0, sigma), eps)


def generated_quantities(*, n, C, year, year_squared, year_cubed, alpha,
                            beta1, beta2, beta3, eps, sigma):
    # Transformed parameters
    log_lambda = alpha + beta1 * year + beta2 * year_squared + beta3 * year_cubed + eps
    # Generated quantities
    lambda__ = exp_vector(log_lambda)
    return { 'log_lambda': log_lambda, 'lambda': lambda__ }

def map_generated_quantities(_samples, *, n, C, year, year_squared,
                                          year_cubed):
    def _generated_quantities(alpha, beta1, beta2, beta3, eps, sigma):
        return generated_quantities(n=n, C=C, year=year,
                                    year_squared=year_squared,
                                    year_cubed=year_cubed, alpha=alpha,
                                    beta1=beta1, beta2=beta2, beta3=beta3,
                                    eps=eps, sigma=sigma)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['alpha'], _samples['beta1'], _samples['beta2'],
              _samples['beta3'], _samples['eps'], _samples['sigma'])

def parameters_info(*, n, C, year, year_squared, year_cubed):
    return { 'alpha': { 'shape': [] },'beta1': { 'shape': [] },
             'beta2': { 'shape': [] },'beta3': { 'shape': [] },
             'eps': { 'shape': [n] },'sigma': { 'shape': [] }, }

