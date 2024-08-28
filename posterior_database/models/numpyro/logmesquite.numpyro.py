from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import log_vector

def convert_inputs(inputs):
    N = inputs['N']
    weight = array(inputs['weight'], dtype=dtype_float)
    diam1 = array(inputs['diam1'], dtype=dtype_float)
    diam2 = array(inputs['diam2'], dtype=dtype_float)
    canopy_height = array(inputs['canopy_height'], dtype=dtype_float)
    total_height = array(inputs['total_height'], dtype=dtype_float)
    density = array(inputs['density'], dtype=dtype_float)
    group = array(inputs['group'], dtype=dtype_float)
    return { 'N': N, 'weight': weight, 'diam1': diam1, 'diam2': diam2,
             'canopy_height': canopy_height, 'total_height': total_height,
             'density': density, 'group': group }

def transformed_data(*, N, weight, diam1, diam2, canopy_height, total_height,
                        density, group):
    # Transformed data
    log_weight = log_vector(weight)
    log_diam1 = log_vector(diam1)
    log_diam2 = log_vector(diam2)
    log_canopy_height = log_vector(canopy_height)
    log_total_height = log_vector(total_height)
    log_density = log_vector(density)
    return { 'log_weight': log_weight, 'log_diam1': log_diam1,
             'log_diam2': log_diam2, 'log_canopy_height': log_canopy_height,
             'log_total_height': log_total_height, 'log_density': log_density }

def model(*, N, weight, diam1, diam2, canopy_height, total_height, density,
             group, log_weight, log_diam1, log_diam2, log_canopy_height,
             log_total_height, log_density):
    # Parameters
    beta__ = sample('beta', improper_uniform(shape=[7]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    # Model
    observe('_log_weight__1', normal(beta__[1 - 1] + beta__[2 - 1] * log_diam1 + beta__[
                                     3 - 1] * log_diam2 + beta__[4 - 1] * log_canopy_height + beta__[
                                     5 - 1] * log_total_height + beta__[
                                     6 - 1] * log_density + beta__[7 - 1] * group,
                                     sigma), log_weight)

def parameters_info(*, N, weight, diam1, diam2, canopy_height, total_height,
                       density, group, log_weight, log_diam1, log_diam2,
                       log_canopy_height, log_total_height, log_density):
    return { 'beta': { 'shape': [7] },'sigma': { 'shape': [] }, }

