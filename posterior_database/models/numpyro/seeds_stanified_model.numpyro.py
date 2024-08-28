from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element

def convert_inputs(inputs):
    I = inputs['I']
    n = array(inputs['n'], dtype=dtype_long)
    N = array(inputs['N'], dtype=dtype_long)
    x1 = array(inputs['x1'], dtype=dtype_float)
    x2 = array(inputs['x2'], dtype=dtype_float)
    return { 'I': I, 'n': n, 'N': N, 'x1': x1, 'x2': x2 }

def transformed_data(*, I, n, N, x1, x2):
    # Transformed data
    x1x2 = x1 * x2
    return { 'x1x2': x1x2 }

def model(*, I, n, N, x1, x2, x1x2):
    # Parameters
    alpha0 = sample('alpha0', improper_uniform(shape=[]))
    alpha1 = sample('alpha1', improper_uniform(shape=[]))
    alpha12 = sample('alpha12', improper_uniform(shape=[]))
    alpha2 = sample('alpha2', improper_uniform(shape=[]))
    b = sample('b', improper_uniform(shape=[I]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    # Model
    observe('_alpha0__1', normal(array(0.0, dtype=dtype_float),
                                 array(1.0, dtype=dtype_float)), alpha0)
    observe('_alpha1__2', normal(array(0.0, dtype=dtype_float),
                                 array(1.0, dtype=dtype_float)), alpha1)
    observe('_alpha2__3', normal(array(0.0, dtype=dtype_float),
                                 array(1.0, dtype=dtype_float)), alpha2)
    observe('_alpha12__4', normal(array(0.0, dtype=dtype_float),
                                  array(1.0, dtype=dtype_float)), alpha12)
    observe('_sigma__5', cauchy(0, 1), sigma)
    observe('_b__6', normal(array(0.0, dtype=dtype_float), sigma), b)
    observe('_n__7', binomial_logit(N,
                                    alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1x2 + b), n)


def parameters_info(*, I, n, N, x1, x2, x1x2):
    return { 'alpha0': { 'shape': [] },'alpha1': { 'shape': [] },
             'alpha12': { 'shape': [] },'alpha2': { 'shape': [] },
             'b': { 'shape': [I] },'sigma': { 'shape': [] }, }

