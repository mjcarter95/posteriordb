from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element


def convert_inputs(inputs):
    N = inputs['N']
    kid_score = array(inputs['kid_score'], dtype=dtype_float)
    mom_iq = array(inputs['mom_iq'], dtype=dtype_float)
    return {'N': N, 'kid_score': kid_score, 'mom_iq': mom_iq}


def model(*, N, kid_score, mom_iq):
    beta__ = sample('beta', improper_uniform(shape=[2]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    observe('_sigma__1', cauchy(0, array(2.5, dtype=dtype_float)), sigma)
    observe('_kid_score__2', normal(beta__[1 - 1] + beta__[2 - 1] * mom_iq,
        sigma), kid_score)


def parameters_info(*, N, kid_score, mom_iq):
    return {'beta': {'shape': [2]}, 'sigma': {'shape': []}}
