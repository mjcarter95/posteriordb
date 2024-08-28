from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element

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

def model(*, N, weight, diam1, diam2, canopy_height, total_height, density,
             group):
    # Parameters
    beta__ = sample('beta', improper_uniform(shape=[7]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    # Model
    observe('_weight__1', normal(beta__[1 - 1] + beta__[2 - 1] * diam1 + beta__[
                                 3 - 1] * diam2 + beta__[4 - 1] * canopy_height + beta__[
                                 5 - 1] * total_height + beta__[6 - 1] * density + beta__[
                                 7 - 1] * group, sigma), weight)

def parameters_info(*, N, weight, diam1, diam2, canopy_height, total_height,
                       density, group):
    return { 'beta': { 'shape': [7] },'sigma': { 'shape': [] }, }

