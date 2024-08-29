from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import append_row_vector_vector, cols_matrix, cumulative_sum_vector, max_vector, mean_vector, min_vector, rep_array_int_int, rep_vector_real_int, rows_matrix, rows_vector, sd_vector, segment_vector_int_int, softmax_vector, sum_array, sum_vector, to_vector_rowvector


def pcm(y, theta, beta__):
    unsummed = append_row_vector_vector(rep_vector_real_int(array(0.0,
        dtype=dtype_float), 1), theta - beta__)
    probs = softmax_vector(cumulative_sum_vector(unsummed))
    return categorical_lpmf(y + 1, probs)


def obtain_adjustments(W):
    adj = empty([2, cols_matrix(W)], dtype=dtype_float)
    adj = adj.at[1 - 1, 1 - 1].set(0)
    adj = adj.at[2 - 1, 1 - 1].set(1)
    min_w = None
    max_w = None
    minmax_count = None

    @jit
    def _then__1(_acc__2):
        adj, max_w, min_w, minmax_count = _acc__2

        @jit
        def _fori__4(k, _acc__5):
            adj, max_w, min_w, minmax_count = _acc__5
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
                adj = adj.at[1 - 1, k - 1].set(mean_vector(W[1 - 1:
                    rows_matrix(W), k - 1]))
                adj = adj.at[2 - 1, k - 1].set(max_w - min_w)
                return adj

            @jit
            def _else__10(_acc__11):
                adj = _acc__11
                adj = adj.at[1 - 1, k - 1].set(mean_vector(W[1 - 1:
                    rows_matrix(W), k - 1]))
                adj = adj.at[2 - 1, k - 1].set(sd_vector(W[1 - 1:
                    rows_matrix(W), k - 1]) * 2)
                return adj
            adj = lax_cond(minmax_count == rows_matrix(W), _then__8,
                _else__10, adj)
            return adj, max_w, min_w, minmax_count
        adj, max_w, min_w, minmax_count = lax_fori_loop(2, cols_matrix(W) +
            1, _fori__4, (adj, max_w, min_w, minmax_count))
        return adj, max_w, min_w, minmax_count

    @jit
    def _else__3(acc):
        return acc
    adj, max_w, min_w, minmax_count = lax_cond(cols_matrix(W) > 1, _then__1,
        _else__3, (adj, max_w, min_w, minmax_count))
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
    return {'I': I, 'N': N, 'ii': ii, 'J': J, 'jj': jj, 'y': y, 'K': K, 'W': W}


def transformed_data(*, I, N, ii, J, jj, y, K, W):
    m = rep_array_int_int(0, I)

    @jit
    def _fori__12(n, _acc__13):
        m = _acc__13

        @jit
        def _then__14(_acc__15):
            m = _acc__15
            m = m.at[ii[n - 1] - 1].set(y[n - 1])
            return m

        @jit
        def _else__16(acc):
            return acc
        m = lax_cond(y[n - 1] > m[ii[n - 1] - 1], _then__14, _else__16, m)
        return m
    m = lax_fori_loop(1, N + 1, _fori__12, m)
    pos = empty([I], dtype=dtype_long)
    pos = pos.at[1 - 1].set(1)

    @jit
    def _fori__17(i, _acc__18):
        pos = _acc__18
        pos = pos.at[i - 1].set(m[i - 1 - 1] + pos[i - 1 - 1])
        return pos
    pos = lax_fori_loop(2, I + 1, _fori__17, pos)
    adj = obtain_adjustments(W)
    W_adj = empty([J, K], dtype=dtype_float)

    @jit
    def _fori__19(k, _acc__20):
        W_adj = _acc__20

        @jit
        def _fori__21(j, _acc__22):
            W_adj = _acc__22
            W_adj = W_adj.at[j - 1, k - 1].set(true_divide(W[j - 1, k - 1] -
                adj[1 - 1, k - 1], adj[2 - 1, k - 1]))
            return W_adj
        W_adj = lax_fori_loop(1, J + 1, _fori__21, W_adj)
        return W_adj
    W_adj = lax_fori_loop(1, K + 1, _fori__19, W_adj)
    return {'m': m, 'pos': pos, 'adj': adj, 'W_adj': W_adj}


def model(*, I, N, ii, J, jj, y, K, W, m, pos, adj, W_adj):
    alpha = sample('alpha', lower_constrained_improper_uniform(0, shape=[I]))
    beta_free = sample('beta_free', improper_uniform(shape=[sum_array(m) - 1]))
    theta = sample('theta', improper_uniform(shape=[J]))
    lambda_adj = sample('lambda_adj', improper_uniform(shape=[K]))
    beta__ = empty([sum_array(m)], dtype=dtype_float)
    beta__ = beta__.at[1 - 1:sum_array(m) - 1].set(beta_free)
    beta__ = beta__.at[sum_array(m) - 1].set(-1 * sum_vector(beta_free))
    observe('_alpha__23', lognormal(1, 1), alpha)
    factor('_expr__24', normal_lpdf(beta__, 0, 3))
    observe('_theta__25', normal(matmul(W_adj, lambda_adj), 1), theta)
    observe('_lambda_adj__26', student_t(3, 0, 1), lambda_adj)

    def _fori__27(n, _acc__28):
        factor(f'_expr__{n}__29', pcm(y[n - 1], theta[jj[n - 1] - 1] *
            alpha[ii[n - 1] - 1], segment_vector_int_int(beta__, pos[ii[n -
            1] - 1], m[ii[n - 1] - 1])))
        return None
    _ = fori_loop(1, N + 1, _fori__27, None)


def generated_quantities(*, I, N, ii, J, jj, y, K, W, m, pos, adj, W_adj,
    alpha, beta_free, theta, lambda_adj):
    beta__ = empty([sum_array(m)], dtype=dtype_float)
    beta__ = beta__.at[1 - 1:sum_array(m) - 1].set(beta_free)
    beta__ = beta__.at[sum_array(m) - 1].set(-1 * sum_vector(beta_free))
    lambda__ = empty([K], dtype=dtype_float)
    lambda__ = lambda__.at[2 - 1:K].set(true_divide(lambda_adj[2 - 1:K],
        to_vector_rowvector(adj[2 - 1, 2 - 1:K])))
    lambda__ = lambda__.at[1 - 1].set(matmul(W_adj[1 - 1, 1 - 1:K],
        lambda_adj[1 - 1:K]) - matmul(W[1 - 1, 2 - 1:K], lambda__[2 - 1:K]))
    return {'beta': beta__, 'lambda': lambda__}


def map_generated_quantities(_samples, *, I, N, ii, J, jj, y, K, W, m, pos,
    adj, W_adj):

    def _generated_quantities(alpha, beta_free, theta, lambda_adj):
        return generated_quantities(I=I, N=N, ii=ii, J=J, jj=jj, y=y, K=K,
            W=W, m=m, pos=pos, adj=adj, W_adj=W_adj, alpha=alpha, beta_free
            =beta_free, theta=theta, lambda_adj=lambda_adj)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['alpha'], _samples['beta_free'], _samples['theta'],
        _samples['lambda_adj'])


def parameters_info(*, I, N, ii, J, jj, y, K, W, m, pos, adj, W_adj):
    return {'alpha': {'shape': [I]}, 'beta_free': {'shape': [sum_array(m) -
        1]}, 'theta': {'shape': [J]}, 'lambda_adj': {'shape': [K]}}
