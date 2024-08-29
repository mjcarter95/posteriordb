from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import log_real, log_sum_exp_array, max_array, negative_infinity


def convert_inputs(inputs):
    K = inputs['K']
    N = inputs['N']
    y = array(inputs['y'], dtype=dtype_float)
    return {'K': K, 'N': N, 'y': y}


def model(*, K, N, y):
    theta1 = sample('theta1', simplex_constrained_improper_uniform(shape=[K]))
    theta2 = sample('theta2', simplex_constrained_improper_uniform(shape=[K]))
    mu = sample('mu', positive_ordered_constrained_improper_uniform(shape=[K]))
    theta = empty([K, K], dtype=dtype_float)
    theta = theta.at[1 - 1].set(theta1)
    theta = theta.at[2 - 1].set(theta2)
    factor('_expr__1', normal_lpdf(mu[1 - 1], 3, 1))
    factor('_expr__2', normal_lpdf(mu[2 - 1], 10, 1))
    gamma__ = empty([N, K], dtype=dtype_float)

    @jit
    def _fori__3(k, _acc__4):
        gamma__ = _acc__4
        gamma__ = gamma__.at[1 - 1, k - 1].set(normal_lpdf(y[1 - 1], mu[k -
            1], 1))
        return gamma__
    gamma__ = lax_fori_loop(1, K + 1, _fori__3, gamma__)
    acc = empty([K], dtype=dtype_float)

    @jit
    def _fori__5(t, _acc__6):
        acc, gamma__ = _acc__6

        @jit
        def _fori__7(k, _acc__8):
            acc, gamma__ = _acc__8

            @jit
            def _fori__9(j, _acc__10):
                acc = _acc__10
                acc = acc.at[j - 1].set(gamma__[t - 1 - 1, j - 1] +
                    log_real(theta[j - 1, k - 1]) + normal_lpdf(y[t - 1],
                    mu[k - 1], 1))
                return acc
            acc = lax_fori_loop(1, K + 1, _fori__9, acc)
            gamma__ = gamma__.at[t - 1, k - 1].set(log_sum_exp_array(acc))
            return acc, gamma__
        acc, gamma__ = lax_fori_loop(1, K + 1, _fori__7, (acc, gamma__))
        return acc, gamma__
    acc, gamma__ = lax_fori_loop(2, N + 1, _fori__5, (acc, gamma__))
    factor('_expr__11', log_sum_exp_array(gamma__[N - 1]))


def generated_quantities(*, K, N, y, theta1, theta2, mu):
    theta = empty([K, K], dtype=dtype_float)
    theta = theta.at[1 - 1].set(theta1)
    theta = theta.at[2 - 1].set(theta2)
    best_logp = empty([N, K], dtype=dtype_float)

    @jit
    def _fori__12(k, _acc__13):
        best_logp = _acc__13
        best_logp = best_logp.at[1 - 1, k - 1].set(normal_lpdf(y[1 - 1], mu
            [k - 1], 1))
        return best_logp
    best_logp = lax_fori_loop(1, K + 1, _fori__12, best_logp)
    back_ptr = empty([N, K], dtype=dtype_long)

    @jit
    def _fori__14(t, _acc__15):
        back_ptr, best_logp = _acc__15

        @jit
        def _fori__16(k, _acc__17):
            back_ptr, best_logp = _acc__17
            best_logp = best_logp.at[t - 1, k - 1].set(negative_infinity())

            @jit
            def _fori__18(j, _acc__19):
                back_ptr, best_logp = _acc__19
                logp = best_logp[t - 1 - 1, j - 1] + log_real(theta[j - 1, 
                    k - 1]) + normal_lpdf(y[t - 1], mu[k - 1], 1)

                @jit
                def _then__20(_acc__21):
                    back_ptr, best_logp = _acc__21
                    back_ptr = back_ptr.at[t - 1, k - 1].set(j)
                    best_logp = best_logp.at[t - 1, k - 1].set(logp)
                    return back_ptr, best_logp

                @jit
                def _else__22(acc):
                    return acc
                back_ptr, best_logp = lax_cond(logp > best_logp[t - 1, k - 
                    1], _then__20, _else__22, (back_ptr, best_logp))
                return back_ptr, best_logp
            back_ptr, best_logp = lax_fori_loop(1, K + 1, _fori__18, (
                back_ptr, best_logp))
            return back_ptr, best_logp
        back_ptr, best_logp = lax_fori_loop(1, K + 1, _fori__16, (back_ptr,
            best_logp))
        return back_ptr, best_logp
    back_ptr, best_logp = lax_fori_loop(2, N + 1, _fori__14, (back_ptr,
        best_logp))
    log_p_z_star = max_array(best_logp[N - 1])
    z_star = empty([N], dtype=dtype_long)

    @jit
    def _fori__23(k, _acc__24):
        z_star = _acc__24

        @jit
        def _then__25(_acc__26):
            z_star = _acc__26
            z_star = z_star.at[N - 1].set(k)
            return z_star

        @jit
        def _else__27(acc):
            return acc
        z_star = lax_cond(best_logp[N - 1, k - 1] == log_p_z_star,
            _then__25, _else__27, z_star)
        return z_star
    z_star = lax_fori_loop(1, K + 1, _fori__23, z_star)

    @jit
    def _fori__28(t, _acc__29):
        z_star = _acc__29
        z_star = z_star.at[N - t - 1].set(back_ptr[N - t + 1 - 1, z_star[N -
            t + 1 - 1] - 1])
        return z_star
    z_star = lax_fori_loop(1, N - 1 + 1, _fori__28, z_star)
    return {'theta': theta, 'best_logp': best_logp, 'back_ptr': back_ptr,
        'log_p_z_star': log_p_z_star, 'z_star': z_star}


def map_generated_quantities(_samples, *, K, N, y):

    def _generated_quantities(theta1, theta2, mu):
        return generated_quantities(K=K, N=N, y=y, theta1=theta1, theta2=
            theta2, mu=mu)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['theta1'], _samples['theta2'], _samples['mu'])


def parameters_info(*, K, N, y):
    return {'theta1': {'shape': [K]}, 'theta2': {'shape': [K]}, 'mu': {
        'shape': [K]}}
