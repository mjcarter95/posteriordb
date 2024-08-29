from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element


def convert_inputs(inputs):
    N = inputs['N']
    encouraged = array(inputs['encouraged'], dtype=dtype_float)
    watched = array(inputs['watched'], dtype=dtype_float)
    return {'N': N, 'encouraged': encouraged, 'watched': watched}


def model(*, N, encouraged, watched):
    beta__ = sample('beta', improper_uniform(shape=[2]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    observe('_watched__1', normal(beta__[1 - 1] + beta__[2 - 1] *
        encouraged, sigma), watched)


def parameters_info(*, N, encouraged, watched):
    return {'beta': {'shape': [2]}, 'sigma': {'shape': []}}
