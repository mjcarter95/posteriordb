from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element


def convert_inputs(inputs):
    N = inputs['N']
    switched = array(inputs['switched'], dtype=dtype_long)
    dist = array(inputs['dist'], dtype=dtype_float)
    return {'N': N, 'switched': switched, 'dist': dist}


def model(*, N, switched, dist):
    beta__ = sample('beta', improper_uniform(shape=[2]))
    observe('_switched__1', bernoulli_logit(beta__[1 - 1] + beta__[2 - 1] *
        dist), switched)


def parameters_info(*, N, switched, dist):
    return {'beta': {'shape': [2]}}
