from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import log_real, log_sum_exp_array, max_array, negative_infinity


def convert_inputs(inputs):
    N = inputs['N']
    u = array(inputs['u'], dtype=dtype_float)
    v = array(inputs['v'], dtype=dtype_float)
    K = inputs['K']
    alpha = array(inputs['alpha'], dtype=dtype_float)
    return {'N': N, 'u': u, 'v': v, 'K': K, 'alpha': alpha}


def model(*, N, u, v, K, alpha):
    theta1 = sample('theta1', simplex_constrained_improper_uniform(shape=[K]))
    theta2 = sample('theta2', simplex_constrained_improper_uniform(shape=[K]))
    phi = sample('phi', positive_ordered_constrained_improper_uniform(shape
        =[K]))
    lambda__ = sample('lambda',
        positive_ordered_constrained_improper_uniform(shape=[K]))
    theta = empty([K, K], dtype=dtype_float)
    theta = theta.at[1 - 1].set(theta1)
    theta = theta.at[2 - 1].set(theta2)

    def _fori__1(k, _acc__2):
        factor(f'_expr__{k}__3', dirichlet_lpdf(theta[k - 1], alpha[k - 1, :]))
        return None
    _ = fori_loop(1, K + 1, _fori__1, None)
    factor('_expr__4', normal_lpdf(phi[1 - 1], 0, 1))
    factor('_expr__5', normal_lpdf(phi[2 - 1], 3, 1))
    factor('_expr__6', normal_lpdf(lambda__[1 - 1], 0, 1))
    factor('_expr__7', normal_lpdf(lambda__[2 - 1], 3, 1))
    gamma__ = empty([N, K], dtype=dtype_float)

    @jit
    def _fori__8(k, _acc__9):
        gamma__ = _acc__9
        gamma__ = gamma__.at[1 - 1, k - 1].set(exponential_lpdf(u[1 - 1],
            phi[k - 1]) + exponential_lpdf(v[1 - 1], lambda__[k - 1]))
        return gamma__
    gamma__ = lax_fori_loop(1, K + 1, _fori__8, gamma__)
    acc = empty([K], dtype=dtype_float)

    @jit
    def _fori__10(t, _acc__11):
        acc, gamma__ = _acc__11

        @jit
        def _fori__12(k, _acc__13):
            acc, gamma__ = _acc__13

            @jit
            def _fori__14(j, _acc__15):
                acc = _acc__15
                acc = acc.at[j - 1].set(gamma__[t - 1 - 1, j - 1] +
                    log_real(theta[j - 1, k - 1]) + exponential_lpdf(u[t - 
                    1], phi[k - 1]) + exponential_lpdf(v[t - 1], lambda__[k -
                    1]))
                return acc
            acc = lax_fori_loop(1, K + 1, _fori__14, acc)
            gamma__ = gamma__.at[t - 1, k - 1].set(log_sum_exp_array(acc))
            return acc, gamma__
        acc, gamma__ = lax_fori_loop(1, K + 1, _fori__12, (acc, gamma__))
        return acc, gamma__
    acc, gamma__ = lax_fori_loop(2, N + 1, _fori__10, (acc, gamma__))
    factor('_expr__16', log_sum_exp_array(gamma__[N - 1]))


def generated_quantities(*, N, u, v, K, alpha, theta1, theta2, phi, lambda__):
    theta = empty([K, K], dtype=dtype_float)
    theta = theta.at[1 - 1].set(theta1)
    theta = theta.at[2 - 1].set(theta2)
    best_logp = empty([N, K], dtype=dtype_float)

    @jit
    def _fori__17(k, _acc__18):
        best_logp = _acc__18
        best_logp = best_logp.at[1 - 1, K - 1].set(exponential_lpdf(u[1 - 1
            ], phi[k - 1]) + exponential_lpdf(v[1 - 1], lambda__[k - 1]))
        return best_logp
    best_logp = lax_fori_loop(1, K + 1, _fori__17, best_logp)
    back_ptr = empty([N, K], dtype=dtype_long)

    @jit
    def _fori__19(t, _acc__20):
        back_ptr, best_logp = _acc__20

        @jit
        def _fori__21(k, _acc__22):
            back_ptr, best_logp = _acc__22
            best_logp = best_logp.at[t - 1, k - 1].set(negative_infinity())

            @jit
            def _fori__23(j, _acc__24):
                back_ptr, best_logp = _acc__24
                logp = best_logp[t - 1 - 1, j - 1] + log_real(theta[j - 1, 
                    k - 1]) + exponential_lpdf(u[t - 1], phi[k - 1]
                    ) + exponential_lpdf(v[t - 1], lambda__[k - 1])

                @jit
                def _then__25(_acc__26):
                    back_ptr, best_logp = _acc__26
                    back_ptr = back_ptr.at[t - 1, k - 1].set(j)
                    best_logp = best_logp.at[t - 1, k - 1].set(logp)
                    return back_ptr, best_logp

                @jit
                def _else__27(acc):
                    return acc
                back_ptr, best_logp = lax_cond(logp > best_logp[t - 1, k - 
                    1], _then__25, _else__27, (back_ptr, best_logp))
                return back_ptr, best_logp
            back_ptr, best_logp = lax_fori_loop(1, K + 1, _fori__23, (
                back_ptr, best_logp))
            return back_ptr, best_logp
        back_ptr, best_logp = lax_fori_loop(1, K + 1, _fori__21, (back_ptr,
            best_logp))
        return back_ptr, best_logp
    back_ptr, best_logp = lax_fori_loop(2, N + 1, _fori__19, (back_ptr,
        best_logp))
    log_p_z_star = max_array(best_logp[N - 1])
    z_star = empty([N], dtype=dtype_long)

    @jit
    def _fori__28(k, _acc__29):
        z_star = _acc__29

        @jit
        def _then__30(_acc__31):
            z_star = _acc__31
            z_star = z_star.at[N - 1].set(k)
            return z_star

        @jit
        def _else__32(acc):
            return acc
        z_star = lax_cond(best_logp[N - 1, k - 1] == log_p_z_star,
            _then__30, _else__32, z_star)
        return z_star
    z_star = lax_fori_loop(1, K + 1, _fori__28, z_star)

    @jit
    def _fori__33(t, _acc__34):
        z_star = _acc__34
        z_star = z_star.at[N - t - 1].set(back_ptr[N - t + 1 - 1, z_star[N -
            t + 1 - 1] - 1])
        return z_star
    z_star = lax_fori_loop(1, N - 1 + 1, _fori__33, z_star)
    return {'theta': theta, 'best_logp': best_logp, 'back_ptr': back_ptr,
        'log_p_z_star': log_p_z_star, 'z_star': z_star}


def map_generated_quantities(_samples, *, N, u, v, K, alpha):

    def _generated_quantities(theta1, theta2, phi, lambda__):
        return generated_quantities(N=N, u=u, v=v, K=K, alpha=alpha, theta1
            =theta1, theta2=theta2, phi=phi, lambda__=lambda__)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['theta1'], _samples['theta2'], _samples['phi'],
        _samples['lambda'])


def parameters_info(*, N, u, v, K, alpha):
    return {'theta1': {'shape': [K]}, 'theta2': {'shape': [K]}, 'phi': {
        'shape': [K]}, 'lambda': {'shape': [K]}}
