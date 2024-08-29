from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element


def convert_inputs(inputs):
    N = inputs['N']
    x = array(inputs['x'], dtype=dtype_float)
    y = array(inputs['y'], dtype=dtype_float)
    xpred = array(inputs['xpred'], dtype=dtype_float)
    pmualpha = array(inputs['pmualpha'], dtype=dtype_float)
    psalpha = array(inputs['psalpha'], dtype=dtype_float)
    pmubeta = array(inputs['pmubeta'], dtype=dtype_float)
    psbeta = array(inputs['psbeta'], dtype=dtype_float)
    return {'N': N, 'x': x, 'y': y, 'xpred': xpred, 'pmualpha': pmualpha,
        'psalpha': psalpha, 'pmubeta': pmubeta, 'psbeta': psbeta}


def model(*, N, x, y, xpred, pmualpha, psalpha, pmubeta, psbeta):
    alpha = sample('alpha', improper_uniform(shape=[]))
    beta__ = sample('beta', improper_uniform(shape=[]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    observe('_alpha__1', normal(pmualpha, psalpha), alpha)
    observe('_beta__2', normal(pmubeta, psbeta), beta__)
    observe('_y__3', normal(alpha + beta__ * x, sigma), y)


def parameters_info(*, N, x, y, xpred, pmualpha, psalpha, pmubeta, psbeta):
    return {'alpha': {'shape': []}, 'beta': {'shape': []}, 'sigma': {
        'shape': []}}
