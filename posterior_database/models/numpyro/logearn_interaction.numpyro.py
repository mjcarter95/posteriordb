from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import log_vector

def convert_inputs(inputs):
    N = inputs['N']
    earn = array(inputs['earn'], dtype=dtype_float)
    height = array(inputs['height'], dtype=dtype_float)
    male = array(inputs['male'], dtype=dtype_float)
    return { 'N': N, 'earn': earn, 'height': height, 'male': male }

def transformed_data(*, N, earn, height, male):
    # Transformed data
    log_earn = log_vector(earn)
    inter = height * male
    return { 'log_earn': log_earn, 'inter': inter }

def model(*, N, earn, height, male, log_earn, inter):
    # Parameters
    beta__ = sample('beta', improper_uniform(shape=[4]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    # Model
    observe('_log_earn__1', normal(beta__[1 - 1] + beta__[2 - 1] * height + beta__[
                                   3 - 1] * male + beta__[4 - 1] * inter,
                                   sigma), log_earn)

def parameters_info(*, N, earn, height, male, log_earn, inter):
    return { 'beta': { 'shape': [4] },'sigma': { 'shape': [] }, }

