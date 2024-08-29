from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import cholesky_decompose_matrix, cols_matrix, dims_array, dot_product_vector_vector, dot_self_vector, exp_real, pi, prod_vector, rep_vector_int_int, rows_vector, size_array, sqrt_real, sqrt_vector, square_real, square_vector


def gp(x, sdgp, lscale, zgp):
    Dls = rows_vector(lscale)
    N = size_array(x)
    cov = empty([N, N], dtype=dtype_float)

    @jit
    def _then__1(_acc__2):
        cov = _acc__2
        cov = gp_exp_quad_cov_array_real_real(x, sdgp, lscale[1 - 1])
        return cov

    @jit
    def _else__3(_acc__4):
        cov = _acc__4
        cov = gp_exp_quad_cov_array_real_real(x[:, 1 - 1], sdgp, lscale[1 - 1])

        @jit
        def _fori__5(d, _acc__6):
            cov = _acc__6
            cov = cov * gp_exp_quad_cov_array_int_real(x[:, d - 1], 1,
                lscale[d - 1])
            return cov
        cov = lax_fori_loop(2, Dls + 1, _fori__5, cov)
        return cov
    cov = lax_cond(Dls == 1, _then__1, _else__3, cov)

    @jit
    def _fori__7(n, _acc__8):
        cov = _acc__8
        cov = cov.at[n - 1, n - 1].set(cov[n - 1, n - 1] + array(1e-12,
            dtype=dtype_float))
        return cov
    cov = lax_fori_loop(1, N + 1, _fori__7, cov)
    return matmul(cholesky_decompose_matrix(cov), zgp)


def spd_cov_exp_quad(x, sdgp, lscale):
    NB = dims_array(x)[1 - 1]
    D = dims_array(x)[2 - 1]
    Dls = rows_vector(lscale)
    out = empty([NB], dtype=dtype_float)

    @jit
    def _then__9(_acc__10):
        out = _acc__10
        constant = square_real(sdgp) * (sqrt_real(2 * pi()) * lscale[1 - 1]
            ) ** D
        neg_half_lscale2 = -array(0.5, dtype=dtype_float) * square_real(lscale
            [1 - 1])

        @jit
        def _fori__13(m, _acc__14):
            out = _acc__14
            out = out.at[m - 1].set(constant * exp_real(neg_half_lscale2 *
                dot_self_vector(x[m - 1])))
            return out
        out = lax_fori_loop(1, NB + 1, _fori__13, out)
        return out

    @jit
    def _else__11(_acc__12):
        out = _acc__12
        constant = square_real(sdgp) * sqrt_real(2 * pi()) ** D * prod_vector(
            lscale)
        neg_half_lscale2 = -array(0.5, dtype=dtype_float) * square_vector(
            lscale)

        @jit
        def _fori__15(m, _acc__16):
            out = _acc__16
            out = out.at[m - 1].set(constant * exp_real(
                dot_product_vector_vector(neg_half_lscale2, square_vector(x
                [m - 1]))))
            return out
        out = lax_fori_loop(1, NB + 1, _fori__15, out)
        return out
    out = lax_cond(Dls == 1, _then__9, _else__11, out)
    return out


def gpa(X, sdgp, lscale, zgp, slambda):
    diag_spd = sqrt_vector(spd_cov_exp_quad(slambda, sdgp, lscale))
    return matmul(X, diag_spd * zgp)


def convert_inputs(inputs):
    N = inputs['N']
    Y = array(inputs['Y'], dtype=dtype_float)
    Kgp_1 = inputs['Kgp_1']
    NBgp_1 = inputs['NBgp_1']
    Xgp_1 = array(inputs['Xgp_1'], dtype=dtype_float)
    Dgp_1 = inputs['Dgp_1']
    slambda_1 = array(inputs['slambda_1'], dtype=dtype_float)
    Kgp_sigma_1 = inputs['Kgp_sigma_1']
    NBgp_sigma_1 = inputs['NBgp_sigma_1']
    Xgp_sigma_1 = array(inputs['Xgp_sigma_1'], dtype=dtype_float)
    Dgp_sigma_1 = inputs['Dgp_sigma_1']
    slambda_sigma_1 = array(inputs['slambda_sigma_1'], dtype=dtype_float)
    prior_only = inputs['prior_only']
    return {'N': N, 'Y': Y, 'Kgp_1': Kgp_1, 'NBgp_1': NBgp_1, 'Xgp_1':
        Xgp_1, 'Dgp_1': Dgp_1, 'slambda_1': slambda_1, 'Kgp_sigma_1':
        Kgp_sigma_1, 'NBgp_sigma_1': NBgp_sigma_1, 'Xgp_sigma_1':
        Xgp_sigma_1, 'Dgp_sigma_1': Dgp_sigma_1, 'slambda_sigma_1':
        slambda_sigma_1, 'prior_only': prior_only}


def transformed_data(*, N, Y, Kgp_1, NBgp_1, Xgp_1, Dgp_1, slambda_1,
    Kgp_sigma_1, NBgp_sigma_1, Xgp_sigma_1, Dgp_sigma_1, slambda_sigma_1,
    prior_only):
    return {}


def model(*, N, Y, Kgp_1, NBgp_1, Xgp_1, Dgp_1, slambda_1, Kgp_sigma_1,
    NBgp_sigma_1, Xgp_sigma_1, Dgp_sigma_1, slambda_sigma_1, prior_only):
    Intercept = sample('Intercept', improper_uniform(shape=[]))
    sdgp_1 = sample('sdgp_1', lower_constrained_improper_uniform(0, shape=[]))
    lscale_1 = sample('lscale_1', lower_constrained_improper_uniform(0,
        shape=[]))
    zgp_1 = sample('zgp_1', improper_uniform(shape=[NBgp_1]))
    Intercept_sigma = sample('Intercept_sigma', improper_uniform(shape=[]))
    sdgp_sigma_1 = sample('sdgp_sigma_1',
        lower_constrained_improper_uniform(0, shape=[]))
    lscale_sigma_1 = sample('lscale_sigma_1',
        lower_constrained_improper_uniform(0, shape=[]))
    zgp_sigma_1 = sample('zgp_sigma_1', improper_uniform(shape=[NBgp_sigma_1]))
    vsdgp_1 = empty([Kgp_1], dtype=dtype_float)
    vsdgp_1 = vsdgp_1.at[1 - 1].set(sdgp_1)
    vlscale_1 = empty([Kgp_1, 1], dtype=dtype_float)
    vlscale_1 = vlscale_1.at[1 - 1, 1 - 1].set(lscale_1)
    vsdgp_sigma_1 = empty([Kgp_sigma_1], dtype=dtype_float)
    vsdgp_sigma_1 = vsdgp_sigma_1.at[1 - 1].set(sdgp_sigma_1)
    vlscale_sigma_1 = empty([Kgp_sigma_1, 1], dtype=dtype_float)
    vlscale_sigma_1 = vlscale_sigma_1.at[1 - 1, 1 - 1].set(lscale_sigma_1)
    mu = Intercept + rep_vector_int_int(0, N) + gpa(Xgp_1, vsdgp_1[1 - 1],
        vlscale_1[1 - 1], zgp_1, slambda_1)
    sigma = Intercept_sigma + rep_vector_int_int(0, N) + gpa(Xgp_sigma_1,
        vsdgp_sigma_1[1 - 1], vlscale_sigma_1[1 - 1], zgp_sigma_1,
        slambda_sigma_1)

    @jit
    def _fori__17(n, _acc__18):
        sigma = _acc__18
        sigma = sigma.at[n - 1].set(exp_real(sigma[n - 1]))
        return sigma
    sigma = lax_fori_loop(1, N + 1, _fori__17, sigma)
    factor('_expr__19', student_t_lpdf(Intercept, 3, -13, 36))
    factor('_expr__20', student_t_lpdf(vsdgp_1, 3, 0, 36) - 1 *
        student_t_lccdf(0, 3, 0, 36))
    factor('_expr__21', normal_lpdf(zgp_1, 0, 1))
    factor('_expr__22', inv_gamma_lpdf(vlscale_1[1 - 1], array(1.124909,
        dtype=dtype_float), array(0.0177, dtype=dtype_float)))
    factor('_expr__23', student_t_lpdf(Intercept_sigma, 3, 0, 10))
    factor('_expr__24', student_t_lpdf(vsdgp_sigma_1, 3, 0, 36) - 1 *
        student_t_lccdf(0, 3, 0, 36))
    factor('_expr__25', normal_lpdf(zgp_sigma_1, 0, 1))
    factor('_expr__26', inv_gamma_lpdf(vlscale_sigma_1[1 - 1], array(
        1.124909, dtype=dtype_float), array(0.0177, dtype=dtype_float)))

    def _then__27(_acc__28):
        factor('_expr__30', normal_lpdf(Y, mu, sigma))
        return None

    def _else__29(acc):
        return acc
    _ = numpyro_cond(not prior_only, _then__27, _else__29, None)


def generated_quantities(*, N, Y, Kgp_1, NBgp_1, Xgp_1, Dgp_1, slambda_1,
    Kgp_sigma_1, NBgp_sigma_1, Xgp_sigma_1, Dgp_sigma_1, slambda_sigma_1,
    prior_only, Intercept, sdgp_1, lscale_1, zgp_1, Intercept_sigma,
    sdgp_sigma_1, lscale_sigma_1, zgp_sigma_1):
    vsdgp_1 = empty([Kgp_1], dtype=dtype_float)
    vsdgp_1 = vsdgp_1.at[1 - 1].set(sdgp_1)
    vlscale_1 = empty([Kgp_1, 1], dtype=dtype_float)
    vlscale_1 = vlscale_1.at[1 - 1, 1 - 1].set(lscale_1)
    vsdgp_sigma_1 = empty([Kgp_sigma_1], dtype=dtype_float)
    vsdgp_sigma_1 = vsdgp_sigma_1.at[1 - 1].set(sdgp_sigma_1)
    vlscale_sigma_1 = empty([Kgp_sigma_1, 1], dtype=dtype_float)
    vlscale_sigma_1 = vlscale_sigma_1.at[1 - 1, 1 - 1].set(lscale_sigma_1)
    b_Intercept = Intercept
    b_sigma_Intercept = Intercept_sigma
    return {'vsdgp_1': vsdgp_1, 'vlscale_1': vlscale_1, 'vsdgp_sigma_1':
        vsdgp_sigma_1, 'vlscale_sigma_1': vlscale_sigma_1, 'b_Intercept':
        b_Intercept, 'b_sigma_Intercept': b_sigma_Intercept}


def map_generated_quantities(_samples, *, N, Y, Kgp_1, NBgp_1, Xgp_1, Dgp_1,
    slambda_1, Kgp_sigma_1, NBgp_sigma_1, Xgp_sigma_1, Dgp_sigma_1,
    slambda_sigma_1, prior_only):

    def _generated_quantities(Intercept, sdgp_1, lscale_1, zgp_1,
        Intercept_sigma, sdgp_sigma_1, lscale_sigma_1, zgp_sigma_1):
        return generated_quantities(N=N, Y=Y, Kgp_1=Kgp_1, NBgp_1=NBgp_1,
            Xgp_1=Xgp_1, Dgp_1=Dgp_1, slambda_1=slambda_1, Kgp_sigma_1=
            Kgp_sigma_1, NBgp_sigma_1=NBgp_sigma_1, Xgp_sigma_1=Xgp_sigma_1,
            Dgp_sigma_1=Dgp_sigma_1, slambda_sigma_1=slambda_sigma_1,
            prior_only=prior_only, Intercept=Intercept, sdgp_1=sdgp_1,
            lscale_1=lscale_1, zgp_1=zgp_1, Intercept_sigma=Intercept_sigma,
            sdgp_sigma_1=sdgp_sigma_1, lscale_sigma_1=lscale_sigma_1,
            zgp_sigma_1=zgp_sigma_1)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['Intercept'], _samples['sdgp_1'], _samples[
        'lscale_1'], _samples['zgp_1'], _samples['Intercept_sigma'],
        _samples['sdgp_sigma_1'], _samples['lscale_sigma_1'], _samples[
        'zgp_sigma_1'])


def parameters_info(*, N, Y, Kgp_1, NBgp_1, Xgp_1, Dgp_1, slambda_1,
    Kgp_sigma_1, NBgp_sigma_1, Xgp_sigma_1, Dgp_sigma_1, slambda_sigma_1,
    prior_only):
    return {'Intercept': {'shape': []}, 'sdgp_1': {'shape': []}, 'lscale_1':
        {'shape': []}, 'zgp_1': {'shape': [NBgp_1]}, 'Intercept_sigma': {
        'shape': []}, 'sdgp_sigma_1': {'shape': []}, 'lscale_sigma_1': {
        'shape': []}, 'zgp_sigma_1': {'shape': [NBgp_sigma_1]}}
