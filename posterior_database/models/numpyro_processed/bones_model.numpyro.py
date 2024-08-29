from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import inv_logit_real, log_real


def convert_inputs(inputs):
    nInd = inputs['nInd']
    gamma__ = array(inputs['gamma'], dtype=dtype_float)
    delta = array(inputs['delta'], dtype=dtype_float)
    ncat = array(inputs['ncat'], dtype=dtype_long)
    nChild = inputs['nChild']
    grade = array(inputs['grade'], dtype=dtype_long)
    return {'nInd': nInd, 'gamma__': gamma__, 'delta': delta, 'ncat': ncat,
        'nChild': nChild, 'grade': grade}


def model(*, nInd, gamma__, delta, ncat, nChild, grade):
    theta = sample('theta', improper_uniform(shape=[nChild]))
    observe('_theta__1', normal(array(0.0, dtype=dtype_float), 36), theta)
    p = empty([nChild, nInd, 5], dtype=dtype_float)
    Q = empty([nChild, nInd, 4], dtype=dtype_float)

    def _fori__2(i, _acc__3):
        Q, p = _acc__3

        def _fori__4(j, _acc__5):
            Q, p = _acc__5

            @jit
            def _fori__6(k, _acc__7):
                Q = _acc__7
                Q = Q.at[i - 1, j - 1, k - 1].set(inv_logit_real(delta[j - 
                    1] * (theta[i - 1] - gamma__[j - 1, k - 1])))
                return Q
            Q = lax_fori_loop(1, ncat[j - 1] - 1 + 1, _fori__6, Q)
            p = p.at[i - 1, j - 1, 1 - 1].set(1 - Q[i - 1, j - 1, 1 - 1])

            @jit
            def _fori__8(k, _acc__9):
                p = _acc__9
                p = p.at[i - 1, j - 1, k - 1].set(Q[i - 1, j - 1, k - 1 - 1
                    ] - Q[i - 1, j - 1, k - 1])
                return p
            p = lax_fori_loop(2, ncat[j - 1] - 1 + 1, _fori__8, p)
            p = p.at[i - 1, j - 1, ncat[j - 1] - 1].set(Q[i - 1, j - 1, 
                ncat[j - 1] - 1 - 1])

            def _then__10(_acc__11):
                factor(f'_expr__{j}__{i}__13', log_real(p[i - 1, j - 1, 
                    grade[i - 1, j - 1] - 1]))
                return None

            def _else__12(acc):
                return acc
            _ = numpyro_cond(grade[i - 1, j - 1] != -1, _then__10,
                _else__12, None)
            return Q, p
        Q, p = fori_loop(1, nInd + 1, _fori__4, (Q, p))
        return Q, p
    Q, p = fori_loop(1, nChild + 1, _fori__2, (Q, p))


def parameters_info(*, nInd, gamma__, delta, ncat, nChild, grade):
    return {'theta': {'shape': [nChild]}}
