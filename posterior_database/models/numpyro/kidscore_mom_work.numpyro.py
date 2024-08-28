from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element

def convert_inputs(inputs):
    N = inputs['N']
    kid_score = array(inputs['kid_score'], dtype=dtype_float)
    mom_work = array(inputs['mom_work'], dtype=dtype_long)
    return { 'N': N, 'kid_score': kid_score, 'mom_work': mom_work }

def transformed_data(*, N, kid_score, mom_work):
    # Transformed data
    work2 = empty([N], dtype=dtype_float)
    work3 = empty([N], dtype=dtype_float)
    work4 = empty([N], dtype=dtype_float)
    @jit
    def _fori__1(i, _acc__2):
        (work2, work3, work4) = _acc__2
        work2 = ops_index_update(work2, ops_index[i - 1], mom_work[i - 1] == 2)
        work3 = ops_index_update(work3, ops_index[i - 1], mom_work[i - 1] == 3)
        work4 = ops_index_update(work4, ops_index[i - 1], mom_work[i - 1] == 4)
        return (work2, work3, work4)
    (work2, work3, work4) = lax_fori_loop(1, N + 1, _fori__1,
                                          (work2, work3, work4))
    return { 'work2': work2, 'work3': work3, 'work4': work4 }

def model(*, N, kid_score, mom_work, work2, work3, work4):
    # Parameters
    beta__ = sample('beta', improper_uniform(shape=[4]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    # Model
    observe('_kid_score__3', normal(beta__[1 - 1] + beta__[2 - 1] * work2 + beta__[
                                    3 - 1] * work3 + beta__[4 - 1] * work4,
                                    sigma), kid_score)

def parameters_info(*, N, kid_score, mom_work, work2, work3, work4):
    return { 'beta': { 'shape': [4] },'sigma': { 'shape': [] }, }

