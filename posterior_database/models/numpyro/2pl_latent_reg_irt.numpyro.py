from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import cols_matrix, max_vector, mean_vector, min_vector, rows_matrix, sd_vector, sum_vector, to_vector_rowvector

def obtain_adjustments(W):
    adj = empty([2, cols_matrix(W)], dtype=dtype_float)
    adj = ops_index_update(adj, ops_index[1 - 1, 1 - 1], 0)
    adj = ops_index_update(adj, ops_index[2 - 1, 1 - 1], 1)
    min_w = None
    max_w = None
    minmax_count = None
    @jit
    def _then__1(_acc__2):
        (adj, max_w, min_w, minmax_count) = _acc__2
        @jit
        def _fori__4(k, _acc__5):
            (adj, max_w, min_w, minmax_count) = _acc__5
            min_w = min_vector(W[1 - 1:rows_matrix(W), k - 1])
            max_w = max_vector(W[1 - 1:rows_matrix(W), k - 1])
            minmax_count = 0
            @jit
            def _fori__6(j, _acc__7):
                minmax_count = _acc__7
                minmax_count = minmax_count + W[j - 1, k - 1] == min_w or W[
                j - 1, k - 1] == max_w
                return minmax_count
            minmax_count = lax_fori_loop(1, rows_matrix(W) + 1, _fori__6,
                                         minmax_count)
            @jit
            def _then__8(_acc__9):
                adj = _acc__9
                adj = ops_index_update(adj, ops_index[1 - 1, k - 1], mean_vector(
                W[1 - 1:rows_matrix(W), k - 1]))
                adj = ops_index_update(adj, ops_index[2 - 1, k - 1], max_w - min_w)
                return adj
            @jit
            def _else__10(_acc__11):
                adj = _acc__11
                adj = ops_index_update(adj, ops_index[1 - 1, k - 1], mean_vector(
                W[1 - 1:rows_matrix(W), k - 1]))
                adj = ops_index_update(adj, ops_index[2 - 1, k - 1], sd_vector(
                W[1 - 1:rows_matrix(W), k - 1]) * 2)
                return adj
            adj = lax_cond(minmax_count == rows_matrix(W),
                           _then__8, _else__10, adj)
            return (adj, max_w, min_w, minmax_count)
        (adj, max_w, min_w, minmax_count) = lax_fori_loop(2,
                                                          cols_matrix(
                                                          W) + 1, _fori__4,
                                                          (adj, max_w, min_w,
                                                           minmax_count))
        return (adj, max_w, min_w, minmax_count)
    @jit
    def _else__3(acc):
        return acc
    (adj, max_w, min_w, minmax_count) = lax_cond(cols_matrix(W) > 1,
                                                 _then__1, _else__3,
                                                 (adj, max_w, min_w,
                                                  minmax_count))
    return adj

def convert_inputs(inputs):
    I = inputs['I']
    N = inputs['N']
    ii = array(inputs['ii'], dtype=dtype_long)
    J = inputs['J']
    jj = array(inputs['jj'], dtype=dtype_long)
    y = array(inputs['y'], dtype=dtype_long)
    K = inputs['K']
    W = array(inputs['W'], dtype=dtype_float)
    return { 'I': I, 'N': N, 'ii': ii, 'J': J, 'jj': jj, 'y': y, 'K': K,
             'W': W }

def transformed_data(*, I, N, ii, J, jj, y, K, W):
    # Transformed data
    adj = obtain_adjustments(W)
    W_adj = empty([J, K], dtype=dtype_float)
    @jit
    def _fori__12(k, _acc__13):
        W_adj = _acc__13
        @jit
        def _fori__14(j, _acc__15):
            W_adj = _acc__15
            W_adj = ops_index_update(W_adj, ops_index[j - 1, k - 1], true_divide((W[
            j - 1, k - 1] - adj[1 - 1, k - 1]), adj[2 - 1, k - 1]))
            return W_adj
        W_adj = lax_fori_loop(1, J + 1, _fori__14, W_adj)
        return W_adj
    W_adj = lax_fori_loop(1, K + 1, _fori__12, W_adj)
    return { 'adj': adj, 'W_adj': W_adj }

def model(*, I, N, ii, J, jj, y, K, W, adj, W_adj):
    # Parameters
    alpha = sample('alpha', lower_constrained_improper_uniform(0, shape=[
    I]))
    beta_free = sample('beta_free', improper_uniform(shape=[I - 1]))
    theta = sample('theta', improper_uniform(shape=[J]))
    lambda_adj = sample('lambda_adj', improper_uniform(shape=[K]))
    # Transformed parameters
    beta__ = empty([I], dtype=dtype_float)
    beta__ = ops_index_update(beta__, ops_index[1 - 1:I - 1], beta_free)
    beta__ = ops_index_update(beta__, ops_index[I - 1], - 1 * sum_vector(
    beta_free))
    # Model
    observe('_alpha__16', lognormal(1, 1), alpha)
    factor('_expr__17', normal_lpdf(beta__, 0, 3))
    observe('_lambda_adj__18', student_t(3, 0, 1), lambda_adj)
    observe('_theta__19', normal(matmul(W_adj, lambda_adj), 1), theta)
    observe('_y__20', bernoulli_logit(alpha[ii - 1] * theta[jj - 1] - beta__[
                                      ii - 1]), y)


def generated_quantities(*, I, N, ii, J, jj, y, K, W, adj, W_adj, alpha,
                            beta_free, theta, lambda_adj):
    # Transformed parameters
    beta__ = empty([I], dtype=dtype_float)
    beta__ = ops_index_update(beta__, ops_index[1 - 1:I - 1], beta_free)
    beta__ = ops_index_update(beta__, ops_index[I - 1], - 1 * sum_vector(
    beta_free))
    # Generated quantities
    lambda__ = empty([K], dtype=dtype_float)
    lambda__ = ops_index_update(lambda__, ops_index[2 - 1:K], true_divide(lambda_adj[
    2 - 1:K], to_vector_rowvector(adj[2 - 1, 2 - 1:K])))
    lambda__ = ops_index_update(lambda__, ops_index[1 - 1], matmul(W_adj[
    1 - 1, 1 - 1:K], lambda_adj[1 - 1:K]) - matmul(W[1 - 1, 2 - 1:K], lambda__[
    2 - 1:K]))
    return { 'beta': beta__, 'lambda': lambda__ }

def map_generated_quantities(_samples, *, I, N, ii, J, jj, y, K, W, adj,
                                          W_adj):
    def _generated_quantities(alpha, beta_free, theta, lambda_adj):
        return generated_quantities(I=I, N=N, ii=ii, J=J, jj=jj, y=y, K=K,
                                    W=W, adj=adj, W_adj=W_adj, alpha=alpha,
                                    beta_free=beta_free, theta=theta,
                                    lambda_adj=lambda_adj)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['alpha'], _samples['beta_free'], _samples['theta'],
              _samples['lambda_adj'])

def parameters_info(*, I, N, ii, J, jj, y, K, W, adj, W_adj):
    return { 'alpha': { 'shape': [I] },'beta_free': { 'shape': [I - 1] },
             'theta': { 'shape': [J] },'lambda_adj': { 'shape': [K] }, }

