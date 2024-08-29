from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element


def convert_inputs(inputs):
    N = inputs['N']
    switched = array(inputs['switched'], dtype=dtype_long)
    dist = array(inputs['dist'], dtype=dtype_float)
    arsenic = array(inputs['arsenic'], dtype=dtype_float)
    educ = array(inputs['educ'], dtype=dtype_float)
    return {'N': N, 'switched': switched, 'dist': dist, 'arsenic': arsenic,
        'educ': educ}


def transformed_data(*, N, switched, dist, arsenic, educ):
    dist100 = true_divide(dist, array(100.0, dtype=dtype_float))
    educ4 = true_divide(educ, array(4.0, dtype=dtype_float))
    x = transpose(array([dist100, arsenic, educ4], dtype=dtype_float), 0, 1)
    return {'dist100': dist100, 'educ4': educ4, 'x': x}


def model(*, N, switched, dist, arsenic, educ, dist100, educ4, x):
    alpha = sample('alpha', improper_uniform(shape=[]))
    beta__ = sample('beta', improper_uniform(shape=[3]))
    observe('_switched__1', bernoulli_logit_glm(x, alpha, beta__), switched)


def parameters_info(*, N, switched, dist, arsenic, educ, dist100, educ4, x):
    return {'alpha': {'shape': []}, 'beta': {'shape': [3]}}
