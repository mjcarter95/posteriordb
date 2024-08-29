from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import append_row_real_vector, cumulative_sum_vector, inv_logit_vector, rep_matrix_int_int_int, rep_row_vector_int_int


def get_changepoint_matrix(t, t_change, T, S):
    A = rep_matrix_int_int_int(0, T, S)
    a_row = rep_row_vector_int_int(0, S)
    cp_idx = 1

    @jit
    def _fori__1(i, _acc__2):
        A, a_row, cp_idx = _acc__2

        @jit
        def _cond__5(_acc__6):
            a_row, cp_idx = _acc__6
            return cp_idx <= S and t[i - 1] >= t_change[cp_idx - 1]

        @jit
        def _while__3(_acc__4):
            a_row, cp_idx = _acc__4
            a_row = a_row.at[cp_idx - 1].set(1)
            cp_idx = cp_idx + 1
            return a_row, cp_idx
        a_row, cp_idx = lax_while_loop(_cond__5, _while__3, (a_row, cp_idx))
        A = A.at[i - 1].set(a_row)
        return A, a_row, cp_idx
    A, a_row, cp_idx = lax_fori_loop(1, T + 1, _fori__1, (A, a_row, cp_idx))
    return A


def logistic_gamma(k, m, delta, t_change, S):
    k_s = append_row_real_vector(k, k + cumulative_sum_vector(delta))
    m_pr = m
    gamma__ = empty([S], dtype=dtype_float)

    @jit
    def _fori__7(i, _acc__8):
        gamma__, m_pr = _acc__8
        gamma__ = gamma__.at[i - 1].set((t_change[i - 1] - m_pr) * (1 -
            true_divide(k_s[i - 1], k_s[i + 1 - 1])))
        m_pr = m_pr + gamma__[i - 1]
        return gamma__, m_pr
    gamma__, m_pr = lax_fori_loop(1, S + 1, _fori__7, (gamma__, m_pr))
    return gamma__


def logistic_trend(k, m, delta, t, cap, A, t_change, S):
    gamma__ = logistic_gamma(k, m, delta, t_change, S)
    return cap * inv_logit_vector((k + matmul(A, delta)) * (t - (m + matmul
        (A, gamma__))))


def linear_trend(k, m, delta, t, A, t_change):
    return (k + matmul(A, delta)) * t + (m + matmul(A, -t_change * delta))


def convert_inputs(inputs):
    T = inputs['T']
    t = array(inputs['t'], dtype=dtype_float)
    cap = array(inputs['cap'], dtype=dtype_float)
    y = array(inputs['y'], dtype=dtype_float)
    S = inputs['S']
    t_change = array(inputs['t_change'], dtype=dtype_float)
    K = inputs['K']
    X = array(inputs['X'], dtype=dtype_float)
    sigmas = array(inputs['sigmas'], dtype=dtype_float)
    tau = array(inputs['tau'], dtype=dtype_float)
    trend_indicator = inputs['trend_indicator']
    s_a = array(inputs['s_a'], dtype=dtype_float)
    s_m = array(inputs['s_m'], dtype=dtype_float)
    return {'T': T, 't': t, 'cap': cap, 'y': y, 'S': S, 't_change':
        t_change, 'K': K, 'X': X, 'sigmas': sigmas, 'tau': tau,
        'trend_indicator': trend_indicator, 's_a': s_a, 's_m': s_m}


def transformed_data(*, T, t, cap, y, S, t_change, K, X, sigmas, tau,
    trend_indicator, s_a, s_m):
    A = get_changepoint_matrix(t, t_change, T, S)
    return {'A': A}


def model(*, T, t, cap, y, S, t_change, K, X, sigmas, tau, trend_indicator,
    s_a, s_m, A):
    k = sample('k', improper_uniform(shape=[]))
    m = sample('m', improper_uniform(shape=[]))
    delta = sample('delta', improper_uniform(shape=[S]))
    sigma_obs = sample('sigma_obs', lower_constrained_improper_uniform(0,
        shape=[]))
    beta__ = sample('beta', improper_uniform(shape=[K]))
    observe('_k__9', normal(0, 5), k)
    observe('_m__10', normal(0, 5), m)
    observe('_delta__11', double_exponential(0, tau), delta)
    observe('_sigma_obs__12', normal(0, array(0.5, dtype=dtype_float)),
        sigma_obs)
    observe('_beta__13', normal(0, sigmas), beta__)

    def _then__14(_acc__15):
        y = _acc__15
        observe('_y__18', normal(linear_trend(k, m, delta, t, A, t_change) *
            (1 + matmul(X, beta__ * s_m)) + matmul(X, beta__ * s_a),
            sigma_obs), y)
        return y

    def _else__16(_acc__17):
        y = _acc__17

        def _then__19(_acc__20):
            y = _acc__20
            observe('_y__22', normal(logistic_trend(k, m, delta, t, cap, A,
                t_change, S) * (1 + matmul(X, beta__ * s_m)) + matmul(X, 
                beta__ * s_a), sigma_obs), y)
            return y

        def _else__21(acc):
            return acc
        y = numpyro_cond(trend_indicator == 1, _then__19, _else__21, y)
        return y
    y = numpyro_cond(trend_indicator == 0, _then__14, _else__16, y)


def parameters_info(*, T, t, cap, y, S, t_change, K, X, sigmas, tau,
    trend_indicator, s_a, s_m, A):
    return {'k': {'shape': []}, 'm': {'shape': []}, 'delta': {'shape': [S]},
        'sigma_obs': {'shape': []}, 'beta': {'shape': [K]}}
