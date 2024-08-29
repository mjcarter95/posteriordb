from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import mean_vector


def convert_inputs(inputs):
    N = inputs['N']
    switched = array(inputs['switched'], dtype=dtype_long)
    dist = array(inputs['dist'], dtype=dtype_float)
    arsenic = array(inputs['arsenic'], dtype=dtype_float)
    educ = array(inputs['educ'], dtype=dtype_float)
    return {'N': N, 'switched': switched, 'dist': dist, 'arsenic': arsenic,
        'educ': educ}


def transformed_data(*, N, switched, dist, arsenic, educ):
    c_dist100 = true_divide(dist - mean_vector(dist), array(100.0, dtype=
        dtype_float))
    c_arsenic = arsenic - mean_vector(arsenic)
    c_educ4 = true_divide(educ - mean_vector(educ), array(4.0, dtype=
        dtype_float))
    da_inter = c_dist100 * c_arsenic
    de_inter = c_dist100 * c_educ4
    ae_inter = c_arsenic * c_educ4
    x = transpose(array([c_dist100, c_arsenic, c_educ4, da_inter, de_inter,
        ae_inter], dtype=dtype_float), 0, 1)
    return {'c_dist100': c_dist100, 'c_arsenic': c_arsenic, 'c_educ4':
        c_educ4, 'da_inter': da_inter, 'de_inter': de_inter, 'ae_inter':
        ae_inter, 'x': x}


def model(*, N, switched, dist, arsenic, educ, c_dist100, c_arsenic,
    c_educ4, da_inter, de_inter, ae_inter, x):
    alpha = sample('alpha', improper_uniform(shape=[]))
    beta__ = sample('beta', improper_uniform(shape=[6]))
    observe('_switched__1', bernoulli_logit_glm(x, alpha, beta__), switched)


def parameters_info(*, N, switched, dist, arsenic, educ, c_dist100,
    c_arsenic, c_educ4, da_inter, de_inter, ae_inter, x):
    return {'alpha': {'shape': []}, 'beta': {'shape': [6]}}
