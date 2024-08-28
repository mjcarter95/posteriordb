from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import inv_logit_real

def convert_inputs(inputs):
    nyears = inputs['nyears']
    C = array(inputs['C'], dtype=dtype_long)
    N = array(inputs['N'], dtype=dtype_long)
    year = array(inputs['year'], dtype=dtype_float)
    return { 'nyears': nyears, 'C': C, 'N': N, 'year': year }

def transformed_data(*, nyears, C, N, year):
    # Transformed data
    year_squared = year * year
    return { 'year_squared': year_squared }

def model(*, nyears, C, N, year, year_squared):
    # Parameters
    alpha = sample('alpha', improper_uniform(shape=[]))
    beta1 = sample('beta1', improper_uniform(shape=[]))
    beta2 = sample('beta2', improper_uniform(shape=[]))
    # Transformed parameters
    logit_p = alpha + beta1 * year + beta2 * year_squared
    # Model
    observe('_alpha__1', normal(0, 100), alpha)
    observe('_beta1__2', normal(0, 100), beta1)
    observe('_beta2__3', normal(0, 100), beta2)
    observe('_C__4', binomial_logit(N, logit_p), C)


def generated_quantities(*, nyears, C, N, year, year_squared, alpha, beta1,
                            beta2):
    # Transformed parameters
    logit_p = alpha + beta1 * year + beta2 * year_squared
    # Generated quantities
    p = empty([nyears], dtype=dtype_float)
    @jit
    def _fori__5(i, _acc__6):
        p = _acc__6
        p = ops_index_update(p, ops_index[i - 1], inv_logit_real(logit_p[
                                                                 i - 1]))
        return p
    p = lax_fori_loop(1, nyears + 1, _fori__5, p)
    return { 'logit_p': logit_p, 'p': p }

def map_generated_quantities(_samples, *, nyears, C, N, year, year_squared):
    def _generated_quantities(alpha, beta1, beta2):
        return generated_quantities(nyears=nyears, C=C, N=N, year=year,
                                    year_squared=year_squared, alpha=alpha,
                                    beta1=beta1, beta2=beta2)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['alpha'], _samples['beta1'], _samples['beta2'])

def parameters_info(*, nyears, C, N, year, year_squared):
    return { 'alpha': { 'shape': [] },'beta1': { 'shape': [] },
             'beta2': { 'shape': [] }, }

