from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element

def convert_inputs(inputs):
    N = inputs['N']
    kid_score = array(inputs['kid_score'], dtype=dtype_float)
    mom_hs = array(inputs['mom_hs'], dtype=dtype_float)
    mom_iq = array(inputs['mom_iq'], dtype=dtype_float)
    return { 'N': N, 'kid_score': kid_score, 'mom_hs': mom_hs,
             'mom_iq': mom_iq }

def transformed_data(*, N, kid_score, mom_hs, mom_iq):
    # Transformed data
    c2_mom_hs = mom_hs - array(0.5, dtype=dtype_float)
    c2_mom_iq = mom_iq - 100
    inter = c2_mom_hs * c2_mom_iq
    return { 'c2_mom_hs': c2_mom_hs, 'c2_mom_iq': c2_mom_iq, 'inter': inter }

def model(*, N, kid_score, mom_hs, mom_iq, c2_mom_hs, c2_mom_iq, inter):
    # Parameters
    beta__ = sample('beta', improper_uniform(shape=[4]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    # Model
    observe('_kid_score__1', normal(beta__[1 - 1] + beta__[2 - 1] * c2_mom_hs + beta__[
                                    3 - 1] * c2_mom_iq + beta__[4 - 1] * inter,
                                    sigma), kid_score)

def parameters_info(*, N, kid_score, mom_hs, mom_iq, c2_mom_hs, c2_mom_iq,
                       inter):
    return { 'beta': { 'shape': [4] },'sigma': { 'shape': [] }, }

