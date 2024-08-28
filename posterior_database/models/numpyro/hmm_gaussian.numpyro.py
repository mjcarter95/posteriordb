from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import log_real, log_sum_exp_array, log_sum_exp_vector, log_vector, max_array, negative_infinity, softmax_vector, sum_vector

def normalize(x):
    return true_divide(x, sum_vector(x))

def convert_inputs(inputs):
    K = inputs['K']
    T = inputs['T']
    y = array(inputs['y'], dtype=dtype_float)
    return { 'K': K, 'T': T, 'y': y }

def model(*, K, T, y):
    # Parameters
    pi1 = sample('pi1', simplex_constrained_improper_uniform(shape=[K]))
    A = sample('A', simplex_constrained_improper_uniform(shape=[K, K]))
    mu = sample('mu', ordered_constrained_improper_uniform(shape=[K]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[
    K]))
    # Transformed parameters
    logalpha = empty([T, K], dtype=dtype_float)
    logalpha = ops_index_update(logalpha, ops_index[1 - 1], log_vector(
    pi1) + normal_lpdf(y[1 - 1], mu, sigma))
    accumulator = empty([K], dtype=dtype_float)
    @jit
    def _fori__1(t, _acc__2):
        (accumulator, logalpha) = _acc__2
        @jit
        def _fori__3(j, _acc__4):
            (accumulator, logalpha) = _acc__4
            @jit
            def _fori__5(i, _acc__6):
                accumulator = _acc__6
                accumulator = ops_index_update(accumulator, ops_index[
                i - 1], logalpha[t - 1 - 1, i - 1] + log_real(A[i - 1, j - 1]) + normal_lpdf(
                y[t - 1], mu[j - 1], sigma[j - 1]))
                return accumulator
            accumulator = lax_fori_loop(1, K + 1, _fori__5, accumulator)
            logalpha = ops_index_update(logalpha, ops_index[t - 1, j - 1], log_sum_exp_array(
            accumulator))
            return (accumulator, logalpha)
        (accumulator, logalpha) = lax_fori_loop(1, K + 1, _fori__3,
                                                (accumulator, logalpha))
        return (accumulator, logalpha)
    (accumulator, logalpha) = lax_fori_loop(2, T + 1, _fori__1,
                                            (accumulator, logalpha))
    # Model
    factor('_expr__7', log_sum_exp_vector(logalpha[T - 1]))


def generated_quantities(*, K, T, y, pi1, A, mu, sigma):
    # Transformed parameters
    logalpha = empty([T, K], dtype=dtype_float)
    logalpha = ops_index_update(logalpha, ops_index[1 - 1], log_vector(
    pi1) + normal_lpdf(y[1 - 1], mu, sigma))
    accumulator = empty([K], dtype=dtype_float)
    @jit
    def _fori__8(t, _acc__9):
        (accumulator, logalpha) = _acc__9
        @jit
        def _fori__10(j, _acc__11):
            (accumulator, logalpha) = _acc__11
            @jit
            def _fori__12(i, _acc__13):
                accumulator = _acc__13
                accumulator = ops_index_update(accumulator, ops_index[
                i - 1], logalpha[t - 1 - 1, i - 1] + log_real(A[i - 1, j - 1]) + normal_lpdf(
                y[t - 1], mu[j - 1], sigma[j - 1]))
                return accumulator
            accumulator = lax_fori_loop(1, K + 1, _fori__12, accumulator)
            logalpha = ops_index_update(logalpha, ops_index[t - 1, j - 1], log_sum_exp_array(
            accumulator))
            return (accumulator, logalpha)
        (accumulator, logalpha) = lax_fori_loop(1, K + 1, _fori__10,
                                                (accumulator, logalpha))
        return (accumulator, logalpha)
    (accumulator, logalpha) = lax_fori_loop(2, T + 1, _fori__8,
                                            (accumulator, logalpha))
    # Generated quantities
    alpha = empty([T, K], dtype=dtype_float)
    @jit
    def _fori__14(t, _acc__15):
        alpha = _acc__15
        alpha = ops_index_update(alpha, ops_index[t - 1], softmax_vector(
        logalpha[t - 1]))
        return alpha
    alpha = lax_fori_loop(1, T + 1, _fori__14, alpha)
    logbeta = empty([T, K], dtype=dtype_float)
    @jit
    def _fori__16(j, _acc__17):
        logbeta = _acc__17
        logbeta = ops_index_update(logbeta, ops_index[T - 1, j - 1], 1)
        return logbeta
    logbeta = lax_fori_loop(1, K + 1, _fori__16, logbeta)
    accumulator = empty([K], dtype=dtype_float)
    @jit
    def _fori__18(tforward, _acc__19):
        (accumulator, logbeta) = _acc__19
        t = T - tforward
        @jit
        def _fori__20(j, _acc__21):
            (accumulator, logbeta) = _acc__21
            @jit
            def _fori__22(i, _acc__23):
                accumulator = _acc__23
                accumulator = ops_index_update(accumulator, ops_index[
                i - 1], logbeta[t - 1, i - 1] + log_real(A[j - 1, i - 1]) + normal_lpdf(
                y[t - 1], mu[i - 1], sigma[i - 1]))
                return accumulator
            accumulator = lax_fori_loop(1, K + 1, _fori__22, accumulator)
            logbeta = ops_index_update(logbeta, ops_index[t - 1 - 1, j - 1], log_sum_exp_array(
            accumulator))
            return (accumulator, logbeta)
        (accumulator, logbeta) = lax_fori_loop(1, K + 1, _fori__20,
                                               (accumulator, logbeta))
        return (accumulator, logbeta)
    (accumulator, logbeta) = lax_fori_loop(0, (T - 2) + 1, _fori__18,
                                           (accumulator, logbeta))
    beta__ = empty([T, K], dtype=dtype_float)
    @jit
    def _fori__24(t, _acc__25):
        beta__ = _acc__25
        beta__ = ops_index_update(beta__, ops_index[t - 1], softmax_vector(
        logbeta[t - 1]))
        return beta__
    beta__ = lax_fori_loop(1, T + 1, _fori__24, beta__)
    loggamma = empty([T, K], dtype=dtype_float)
    @jit
    def _fori__26(t, _acc__27):
        loggamma = _acc__27
        loggamma = ops_index_update(loggamma, ops_index[t - 1], alpha[
        t - 1] * beta__[t - 1])
        return loggamma
    loggamma = lax_fori_loop(1, T + 1, _fori__26, loggamma)
    gamma__ = empty([T, K], dtype=dtype_float)
    @jit
    def _fori__28(t, _acc__29):
        gamma__ = _acc__29
        gamma__ = ops_index_update(gamma__, ops_index[t - 1], normalize(
        loggamma[t - 1]))
        return gamma__
    gamma__ = lax_fori_loop(1, T + 1, _fori__28, gamma__)
    delta = empty([T, K], dtype=dtype_float)
    @jit
    def _fori__30(j, _acc__31):
        delta = _acc__31
        delta = ops_index_update(delta, ops_index[1 - 1, K - 1], normal_lpdf(
        y[1 - 1], mu[j - 1], sigma[j - 1]))
        return delta
    delta = lax_fori_loop(1, K + 1, _fori__30, delta)
    bpointer = empty([T, K], dtype=dtype_long)
    @jit
    def _fori__32(t, _acc__33):
        (bpointer, delta) = _acc__33
        @jit
        def _fori__34(j, _acc__35):
            (bpointer, delta) = _acc__35
            delta = ops_index_update(delta, ops_index[t - 1, j - 1], negative_infinity(
            ))
            @jit
            def _fori__36(i, _acc__37):
                (bpointer, delta) = _acc__37
                logp = delta[t - 1 - 1, i - 1] + log_real(A[i - 1, j - 1]) + normal_lpdf(
                y[t - 1], mu[j - 1], sigma[j - 1])
                @jit
                def _then__38(_acc__39):
                    (bpointer, delta) = _acc__39
                    bpointer = ops_index_update(bpointer, ops_index[t - 1,
                                                                    j - 1], i)
                    delta = ops_index_update(delta, ops_index[t - 1, j - 1], logp)
                    return (bpointer, delta)
                @jit
                def _else__40(acc):
                    return acc
                (bpointer, delta) = lax_cond(logp > delta[t - 1, j - 1],
                                             _then__38, _else__40,
                                             (bpointer, delta))
                return (bpointer, delta)
            (bpointer, delta) = lax_fori_loop(1, K + 1, _fori__36,
                                              (bpointer, delta))
            return (bpointer, delta)
        (bpointer, delta) = lax_fori_loop(1, K + 1, _fori__34,
                                          (bpointer, delta))
        return (bpointer, delta)
    (bpointer, delta) = lax_fori_loop(2, T + 1, _fori__32, (bpointer, delta))
    logp_zstar = max_array(delta[T - 1])
    zstar = empty([T], dtype=dtype_long)
    @jit
    def _fori__41(j, _acc__42):
        zstar = _acc__42
        @jit
        def _then__43(_acc__44):
            zstar = _acc__44
            zstar = ops_index_update(zstar, ops_index[T - 1], j)
            return zstar
        @jit
        def _else__45(acc):
            return acc
        zstar = lax_cond(delta[T - 1, j - 1] == logp_zstar,
                         _then__43, _else__45, zstar)
        return zstar
    zstar = lax_fori_loop(1, K + 1, _fori__41, zstar)
    @jit
    def _fori__46(t, _acc__47):
        zstar = _acc__47
        zstar = ops_index_update(zstar, ops_index[T - t - 1], bpointer[
        T - t + 1 - 1, zstar[T - t + 1 - 1] - 1])
        return zstar
    zstar = lax_fori_loop(1, (T - 1) + 1, _fori__46, zstar)
    return { 'logalpha': logalpha, 'accumulator': accumulator,
             'alpha': alpha, 'logbeta': logbeta, 'accumulator': accumulator,
             'beta': beta__, 'loggamma': loggamma, 'gamma': gamma__,
             'delta': delta, 'bpointer': bpointer, 'logp_zstar': logp_zstar,
             'zstar': zstar }

def map_generated_quantities(_samples, *, K, T, y):
    def _generated_quantities(pi1, A, mu, sigma):
        return generated_quantities(K=K, T=T, y=y, pi1=pi1, A=A, mu=mu,
                                    sigma=sigma)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['pi1'], _samples['A'], _samples['mu'],
              _samples['sigma'])

def parameters_info(*, K, T, y):
    return { 'pi1': { 'shape': [K] },'A': { 'shape': [K, K] },
             'mu': { 'shape': [K] },'sigma': { 'shape': [K] }, }

