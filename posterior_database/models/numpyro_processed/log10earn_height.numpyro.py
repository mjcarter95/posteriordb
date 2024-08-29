from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import log10_real


def convert_inputs(inputs):
    N = inputs['N']
    earn = array(inputs['earn'], dtype=dtype_float)
    height = array(inputs['height'], dtype=dtype_float)
    return {'N': N, 'earn': earn, 'height': height}


def transformed_data(*, N, earn, height):
    log10_earn = empty([N], dtype=dtype_float)

    @jit
    def _fori__1(i, _acc__2):
        log10_earn = _acc__2
        log10_earn = log10_earn.at[i - 1].set(log10_real(earn[i - 1]))
        return log10_earn
    log10_earn = lax_fori_loop(1, N + 1, _fori__1, log10_earn)
    return {'log10_earn': log10_earn}


def model(*, N, earn, height, log10_earn):
    beta__ = sample('beta', improper_uniform(shape=[2]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    observe('_log10_earn__3', normal(beta__[1 - 1] + beta__[2 - 1] * height,
        sigma), log10_earn)


def parameters_info(*, N, earn, height, log10_earn):
    return {'beta': {'shape': [2]}, 'sigma': {'shape': []}}
