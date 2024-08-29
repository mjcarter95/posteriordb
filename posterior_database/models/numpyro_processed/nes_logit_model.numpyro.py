from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element


def convert_inputs(inputs):
    N = inputs['N']
    income = array(inputs['income'], dtype=dtype_float)
    vote = array(inputs['vote'], dtype=dtype_long)
    return {'N': N, 'income': income, 'vote': vote}


def transformed_data(*, N, income, vote):
    x = transpose(array([income], dtype=dtype_float), 0, 1)
    return {'x': x}


def model(*, N, income, vote, x):
    alpha = sample('alpha', improper_uniform(shape=[]))
    beta__ = sample('beta', improper_uniform(shape=[1]))
    observe('_vote__1', bernoulli_logit_glm(x, alpha, beta__), vote)


def parameters_info(*, N, income, vote, x):
    return {'alpha': {'shape': []}, 'beta': {'shape': [1]}}
