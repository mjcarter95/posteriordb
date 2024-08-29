from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import log_real, log_sum_exp_array, log_sum_exp_vector, log_vector, max_array, negative_infinity, softmax_vector, sum_vector, to_vector_vector


def normalize(x):
    return true_divide(x, sum_vector(x))


def convert_inputs(inputs):
    K = inputs['K']
    T = inputs['T']
    y = array(inputs['y'], dtype=dtype_float)
    M = inputs['M']
    u = array(inputs['u'], dtype=dtype_float)
    return {'K': K, 'T': T, 'y': y, 'M': M, 'u': u}


def model(*, K, T, y, M, u):
    pi1 = sample('pi1', simplex_constrained_improper_uniform(shape=[K]))
    w = sample('w', improper_uniform(shape=[K, M]))
    b = sample('b', improper_uniform(shape=[K, M]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[K]))
    unA = empty([T, K], dtype=dtype_float)
    unA = unA.at[1 - 1].set(pi1)
    A = empty([T, K], dtype=dtype_float)
    A = A.at[1 - 1].set(pi1)
    logA = empty([T, K], dtype=dtype_float)
    logA = logA.at[1 - 1].set(log_vector(A[1 - 1]))

    @jit
    def _fori__1(t, _acc__2):
        A, logA, unA = _acc__2

        @jit
        def _fori__3(j, _acc__4):
            unA = _acc__4
            unA = unA.at[ops_index[t - 1], ops_index[j - 1]].set(matmul(u[t -
                1], w[j - 1]))
            return unA
        unA = lax_fori_loop(1, K + 1, _fori__3, unA)
        A = A.at[t - 1].set(softmax_vector(unA[t - 1]))
        logA = logA.at[t - 1].set(log_vector(A[t - 1]))
        return A, logA, unA
    A, logA, unA = lax_fori_loop(2, T + 1, _fori__1, (A, logA, unA))
    logoblik = empty([T, K], dtype=dtype_float)

    @jit
    def _fori__5(t, _acc__6):
        logoblik = _acc__6

        @jit
        def _fori__7(j, _acc__8):
            logoblik = _acc__8
            logoblik = logoblik.at[ops_index[t - 1], ops_index[j - 1]].set(
                normal_lpdf(y[t - 1], matmul(u[t - 1], b[j - 1]), sigma[j - 1])
                )
            return logoblik
        logoblik = lax_fori_loop(1, K + 1, _fori__7, logoblik)
        return logoblik
    logoblik = lax_fori_loop(1, T + 1, _fori__5, logoblik)
    logalpha = empty([T, K], dtype=dtype_float)

    @jit
    def _fori__9(j, _acc__10):
        logalpha = _acc__10
        logalpha = logalpha.at[ops_index[1 - 1], ops_index[j - 1]].set(
            log_real(pi1[j - 1]) + logoblik[1 - 1][j - 1])
        return logalpha
    logalpha = lax_fori_loop(1, K + 1, _fori__9, logalpha)
    accumulator = empty([K], dtype=dtype_float)

    @jit
    def _fori__11(t, _acc__12):
        accumulator, logalpha = _acc__12

        @jit
        def _fori__13(j, _acc__14):
            accumulator, logalpha = _acc__14

            @jit
            def _fori__15(i, _acc__16):
                accumulator = _acc__16
                accumulator = accumulator.at[i - 1].set(logalpha[t - 1 - 1,
                    i - 1] + logA[t - 1][i - 1] + logoblik[t - 1][j - 1])
                return accumulator
            accumulator = lax_fori_loop(1, K + 1, _fori__15, accumulator)
            logalpha = logalpha.at[t - 1, j - 1].set(log_sum_exp_array(
                accumulator))
            return accumulator, logalpha
        accumulator, logalpha = lax_fori_loop(1, K + 1, _fori__13, (
            accumulator, logalpha))
        return accumulator, logalpha
    accumulator, logalpha = lax_fori_loop(2, T + 1, _fori__11, (accumulator,
        logalpha))

    def _fori__17(j, _acc__18):
        observe(f'_w__{j}__19', normal(0, 5), w[j - 1])
        observe(f'_b__{j}__20', normal(0, 5), b[j - 1])
        observe(f'_sigma__{j}__21', normal(0, 3), sigma[j - 1])
        return None
    _ = fori_loop(1, K + 1, _fori__17, None)
    factor('_expr__22', log_sum_exp_vector(logalpha[T - 1]))


def generated_quantities(*, K, T, y, M, u, pi1, w, b, sigma):
    unA = empty([T, K], dtype=dtype_float)
    unA = unA.at[1 - 1].set(pi1)
    A = empty([T, K], dtype=dtype_float)
    A = A.at[1 - 1].set(pi1)
    logA = empty([T, K], dtype=dtype_float)
    logA = logA.at[1 - 1].set(log_vector(A[1 - 1]))

    @jit
    def _fori__23(t, _acc__24):
        A, logA, unA = _acc__24

        @jit
        def _fori__25(j, _acc__26):
            unA = _acc__26
            unA = unA.at[ops_index[t - 1], ops_index[j - 1]].set(matmul(u[t -
                1], w[j - 1]))
            return unA
        unA = lax_fori_loop(1, K + 1, _fori__25, unA)
        A = A.at[t - 1].set(softmax_vector(unA[t - 1]))
        logA = logA.at[t - 1].set(log_vector(A[t - 1]))
        return A, logA, unA
    A, logA, unA = lax_fori_loop(2, T + 1, _fori__23, (A, logA, unA))
    logoblik = empty([T, K], dtype=dtype_float)

    @jit
    def _fori__27(t, _acc__28):
        logoblik = _acc__28

        @jit
        def _fori__29(j, _acc__30):
            logoblik = _acc__30
            logoblik = logoblik.at[ops_index[t - 1], ops_index[j - 1]].set(
                normal_lpdf(y[t - 1], matmul(u[t - 1], b[j - 1]), sigma[j - 1])
                )
            return logoblik
        logoblik = lax_fori_loop(1, K + 1, _fori__29, logoblik)
        return logoblik
    logoblik = lax_fori_loop(1, T + 1, _fori__27, logoblik)
    logalpha = empty([T, K], dtype=dtype_float)

    @jit
    def _fori__31(j, _acc__32):
        logalpha = _acc__32
        logalpha = logalpha.at[ops_index[1 - 1], ops_index[j - 1]].set(
            log_real(pi1[j - 1]) + logoblik[1 - 1][j - 1])
        return logalpha
    logalpha = lax_fori_loop(1, K + 1, _fori__31, logalpha)
    accumulator = empty([K], dtype=dtype_float)

    @jit
    def _fori__33(t, _acc__34):
        accumulator, logalpha = _acc__34

        @jit
        def _fori__35(j, _acc__36):
            accumulator, logalpha = _acc__36

            @jit
            def _fori__37(i, _acc__38):
                accumulator = _acc__38
                accumulator = accumulator.at[i - 1].set(logalpha[t - 1 - 1,
                    i - 1] + logA[t - 1][i - 1] + logoblik[t - 1][j - 1])
                return accumulator
            accumulator = lax_fori_loop(1, K + 1, _fori__37, accumulator)
            logalpha = logalpha.at[t - 1, j - 1].set(log_sum_exp_array(
                accumulator))
            return accumulator, logalpha
        accumulator, logalpha = lax_fori_loop(1, K + 1, _fori__35, (
            accumulator, logalpha))
        return accumulator, logalpha
    accumulator, logalpha = lax_fori_loop(2, T + 1, _fori__33, (accumulator,
        logalpha))
    alpha = empty([T, K], dtype=dtype_float)

    @jit
    def _fori__39(t, _acc__40):
        alpha = _acc__40
        alpha = alpha.at[t - 1].set(softmax_vector(logalpha[t - 1]))
        return alpha
    alpha = lax_fori_loop(1, T + 1, _fori__39, alpha)
    logbeta = empty([T, K], dtype=dtype_float)

    @jit
    def _fori__41(j, _acc__42):
        logbeta = _acc__42
        logbeta = logbeta.at[T - 1, j - 1].set(1)
        return logbeta
    logbeta = lax_fori_loop(1, K + 1, _fori__41, logbeta)
    accumulator = empty([K], dtype=dtype_float)
    tbackwards = None

    @jit
    def _fori__43(tforwards, _acc__44):
        accumulator, logbeta, tbackwards = _acc__44
        tbackwards = T - tforwards

        @jit
        def _fori__45(j, _acc__46):
            accumulator, logbeta = _acc__46

            @jit
            def _fori__47(i, _acc__48):
                accumulator = _acc__48
                accumulator = accumulator.at[i - 1].set(logbeta[tbackwards -
                    1, i - 1] + logA[tbackwards - 1][i - 1] + logoblik[
                    tbackwards - 1][i - 1])
                return accumulator
            accumulator = lax_fori_loop(1, K + 1, _fori__47, accumulator)
            logbeta = logbeta.at[tbackwards - 1 - 1, j - 1].set(
                log_sum_exp_array(accumulator))
            return accumulator, logbeta
        accumulator, logbeta = lax_fori_loop(1, K + 1, _fori__45, (
            accumulator, logbeta))
        return accumulator, logbeta, tbackwards
    accumulator, logbeta, tbackwards = lax_fori_loop(0, T - 2 + 1,
        _fori__43, (accumulator, logbeta, tbackwards))
    beta__ = empty([T, K], dtype=dtype_float)

    @jit
    def _fori__49(t, _acc__50):
        beta__ = _acc__50
        beta__ = beta__.at[t - 1].set(softmax_vector(logbeta[t - 1]))
        return beta__
    beta__ = lax_fori_loop(1, T + 1, _fori__49, beta__)
    loggamma = empty([T, K], dtype=dtype_float)

    @jit
    def _fori__51(t, _acc__52):
        loggamma = _acc__52
        loggamma = loggamma.at[t - 1].set(alpha[t - 1] * beta__[t - 1])
        return loggamma
    loggamma = lax_fori_loop(1, T + 1, _fori__51, loggamma)
    gamma__ = empty([T, K], dtype=dtype_float)

    @jit
    def _fori__53(t, _acc__54):
        gamma__ = _acc__54
        gamma__ = gamma__.at[t - 1].set(normalize(loggamma[t - 1]))
        return gamma__
    gamma__ = lax_fori_loop(1, T + 1, _fori__53, gamma__)
    hatpi = empty([T, K], dtype=dtype_float)
    hatz = empty([T], dtype=dtype_long)
    reg = empty([T, K], dtype=dtype_float)

    @jit
    def _fori__55(t, _acc__56):
        hatpi, hatz, reg = _acc__56

        @jit
        def _fori__57(j, _acc__58):
            reg = _acc__58
            reg = reg.at[t - 1, j - 1].set(matmul(u[t - 1],
                to_vector_vector(w[j - 1])))
            return reg
        reg = lax_fori_loop(1, K + 1, _fori__57, reg)
        hatpi = hatpi.at[t - 1].set(softmax_vector(reg[t - 1]))
        hatz = hatz.at[t - 1].set(categorical_rng(hatpi[t - 1]))
        return hatpi, hatz, reg
    hatpi, hatz, reg = lax_fori_loop(1, T + 1, _fori__55, (hatpi, hatz, reg))
    haty = empty([T], dtype=dtype_float)
    reg = empty([T], dtype=dtype_float)

    @jit
    def _fori__59(t, _acc__60):
        haty, reg = _acc__60
        reg = reg.at[t - 1].set(matmul(u[t - 1], b[hatz[t - 1] - 1]))
        haty = haty.at[t - 1].set(normal_rng(reg[t - 1], sigma[hatz[t - 1] -
            1]))
        return haty, reg
    haty, reg = lax_fori_loop(1, T + 1, _fori__59, (haty, reg))
    delta = empty([T, K], dtype=dtype_float)

    @jit
    def _fori__61(j, _acc__62):
        delta = _acc__62
        delta = delta.at[1 - 1, K - 1].set(logoblik[1 - 1][j - 1])
        return delta
    delta = lax_fori_loop(1, K + 1, _fori__61, delta)
    bpointer = empty([T, K], dtype=dtype_long)

    @jit
    def _fori__63(t, _acc__64):
        bpointer, delta = _acc__64

        @jit
        def _fori__65(j, _acc__66):
            bpointer, delta = _acc__66
            delta = delta.at[t - 1, j - 1].set(negative_infinity())

            @jit
            def _fori__67(i, _acc__68):
                bpointer, delta = _acc__68
                logp = delta[t - 1 - 1, i - 1] + logA[t - 1][i - 1] + logoblik[
                    t - 1][j - 1]

                @jit
                def _then__69(_acc__70):
                    bpointer, delta = _acc__70
                    bpointer = bpointer.at[t - 1, j - 1].set(i)
                    delta = delta.at[t - 1, j - 1].set(logp)
                    return bpointer, delta

                @jit
                def _else__71(acc):
                    return acc
                bpointer, delta = lax_cond(logp > delta[t - 1, j - 1],
                    _then__69, _else__71, (bpointer, delta))
                return bpointer, delta
            bpointer, delta = lax_fori_loop(1, K + 1, _fori__67, (bpointer,
                delta))
            return bpointer, delta
        bpointer, delta = lax_fori_loop(1, K + 1, _fori__65, (bpointer, delta))
        return bpointer, delta
    bpointer, delta = lax_fori_loop(2, T + 1, _fori__63, (bpointer, delta))
    logp_zstar = max_array(delta[T - 1])
    zstar = empty([T], dtype=dtype_long)

    @jit
    def _fori__72(j, _acc__73):
        zstar = _acc__73

        @jit
        def _then__74(_acc__75):
            zstar = _acc__75
            zstar = zstar.at[T - 1].set(j)
            return zstar

        @jit
        def _else__76(acc):
            return acc
        zstar = lax_cond(delta[T - 1, j - 1] == logp_zstar, _then__74,
            _else__76, zstar)
        return zstar
    zstar = lax_fori_loop(1, K + 1, _fori__72, zstar)

    @jit
    def _fori__77(t, _acc__78):
        zstar = _acc__78
        zstar = zstar.at[T - t - 1].set(bpointer[T - t + 1 - 1, zstar[T - t +
            1 - 1] - 1])
        return zstar
    zstar = lax_fori_loop(1, T - 1 + 1, _fori__77, zstar)
    return {'unA': unA, 'A': A, 'logA': logA, 'logoblik': logoblik,
        'logalpha': logalpha, 'accumulator': accumulator, 'alpha': alpha,
        'logbeta': logbeta, 'accumulator': accumulator, 'tbackwards':
        tbackwards, 'beta': beta__, 'loggamma': loggamma, 'gamma': gamma__,
        'hatpi': hatpi, 'hatz': hatz, 'reg': reg, 'haty': haty, 'reg': reg,
        'delta': delta, 'bpointer': bpointer, 'logp_zstar': logp_zstar,
        'zstar': zstar}


def map_generated_quantities(_samples, *, K, T, y, M, u):

    def _generated_quantities(pi1, w, b, sigma):
        return generated_quantities(K=K, T=T, y=y, M=M, u=u, pi1=pi1, w=w,
            b=b, sigma=sigma)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['pi1'], _samples['w'], _samples['b'], _samples['sigma'])


def parameters_info(*, K, T, y, M, u):
    return {'pi1': {'shape': [K]}, 'w': {'shape': [K, M]}, 'b': {'shape': [
        K, M]}, 'sigma': {'shape': [K]}}
