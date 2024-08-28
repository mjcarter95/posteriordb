from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import append_row_vector_vector, cols_matrix, cumulative_sum_vector, max_array, max_vector, mean_vector, min_vector, rep_vector_int_int, rep_vector_real_int, rows_matrix, rows_vector, sd_vector, softmax_vector, sum_vector

def rsm(y, theta, beta__, kappa):
    unsummed = append_row_vector_vector(rep_vector_int_int(0, 1),
                                        theta - beta__ - kappa)
    probs = softmax_vector(cumulative_sum_vector(unsummed))
    return categorical_lpmf(y + 1, probs)

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
    m = max_array(y)
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
    return { 'm': m, 'adj': adj, 'W_adj': W_adj }

def model(*, I, N, ii, J, jj, y, K, W, m, adj, W_adj):
    # Parameters
    alpha = sample('alpha', lower_constrained_improper_uniform(0, shape=[
    I]))
    beta_free = sample('beta_free', improper_uniform(shape=[I - 1]))
    kappa_free = sample('kappa_free', improper_uniform(shape=[m - 1]))
    theta = sample('theta', improper_uniform(shape=[J]))
    lambda_adj = sample('lambda_adj', improper_uniform(shape=[K]))
    # Transformed parameters
    beta__ = append_row_vector_vector(beta_free,
                                      rep_vector_real_int(- 1 * sum_vector(
                                                          beta_free), 1))
    kappa = append_row_vector_vector(kappa_free,
                                     rep_vector_real_int(- 1 * sum_vector(
                                                         kappa_free), 1))
    # Model
    observe('_alpha__16', lognormal(1, 1), alpha)
    factor('_expr__17', normal_lpdf(beta__, 0, 3))
    factor('_expr__18', normal_lpdf(kappa, 0, 3))
    observe('_theta__19', normal(matmul(W_adj, lambda_adj), 1), theta)
    observe('_lambda_adj__20', student_t(3, 0, 1), lambda_adj)
    def _fori__21(n, _acc__22):
        factor(f'_expr__{n}__23', rsm(y[n - 1],
                                      theta[jj[n - 1] - 1] * alpha[ii[
                                                                   n - 1] - 1],
                                      beta__[ii[n - 1] - 1], kappa))
        return None
    _ = fori_loop(1, N + 1, _fori__21, None)


def generated_quantities(*, I, N, ii, J, jj, y, K, W, m, adj, W_adj, alpha,
                            beta_free, kappa_free, theta, lambda_adj):
    # Transformed parameters
    beta__ = append_row_vector_vector(beta_free,
                                      rep_vector_real_int(- 1 * sum_vector(
                                                          beta_free), 1))
    kappa = append_row_vector_vector(kappa_free,
                                     rep_vector_real_int(- 1 * sum_vector(
                                                         kappa_free), 1))
    # Generated quantities
    
    return { 'beta': beta__, 'kappa': kappa }

def map_generated_quantities(_samples, *, I, N, ii, J, jj, y, K, W, m, adj,
                                          W_adj):
    def _generated_quantities(alpha, beta_free, kappa_free, theta, lambda_adj):
        return generated_quantities(I=I, N=N, ii=ii, J=J, jj=jj, y=y, K=K,
                                    W=W, m=m, adj=adj, W_adj=W_adj,
                                    alpha=alpha, beta_free=beta_free,
                                    kappa_free=kappa_free, theta=theta,
                                    lambda_adj=lambda_adj)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['alpha'], _samples['beta_free'],
              _samples['kappa_free'], _samples['theta'],
              _samples['lambda_adj'])

def parameters_info(*, I, N, ii, J, jj, y, K, W, m, adj, W_adj):
    return { 'alpha': { 'shape': [I] },'beta_free': { 'shape': [I - 1] },
             'kappa_free': { 'shape': [m - 1] },'theta': { 'shape': [
             J] },'lambda_adj': { 'shape': [K] }, }

