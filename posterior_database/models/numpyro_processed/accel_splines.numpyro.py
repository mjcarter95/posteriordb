from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import exp_real, rep_vector_int_int


def convert_inputs(inputs):
    N = inputs['N']
    Y = array(inputs['Y'], dtype=dtype_float)
    Ks = inputs['Ks']
    Xs = array(inputs['Xs'], dtype=dtype_float)
    knots_1 = inputs['knots_1']
    Zs_1_1 = array(inputs['Zs_1_1'], dtype=dtype_float)
    Ks_sigma = inputs['Ks_sigma']
    Xs_sigma = array(inputs['Xs_sigma'], dtype=dtype_float)
    knots_sigma_1 = inputs['knots_sigma_1']
    Zs_sigma_1_1 = array(inputs['Zs_sigma_1_1'], dtype=dtype_float)
    prior_only = inputs['prior_only']
    return {'N': N, 'Y': Y, 'Ks': Ks, 'Xs': Xs, 'knots_1': knots_1,
        'Zs_1_1': Zs_1_1, 'Ks_sigma': Ks_sigma, 'Xs_sigma': Xs_sigma,
        'knots_sigma_1': knots_sigma_1, 'Zs_sigma_1_1': Zs_sigma_1_1,
        'prior_only': prior_only}


def transformed_data(*, N, Y, Ks, Xs, knots_1, Zs_1_1, Ks_sigma, Xs_sigma,
    knots_sigma_1, Zs_sigma_1_1, prior_only):
    return {}


def model(*, N, Y, Ks, Xs, knots_1, Zs_1_1, Ks_sigma, Xs_sigma,
    knots_sigma_1, Zs_sigma_1_1, prior_only):
    Intercept = sample('Intercept', improper_uniform(shape=[]))
    bs = sample('bs', improper_uniform(shape=[Ks]))
    zs_1_1 = sample('zs_1_1', improper_uniform(shape=[knots_1]))
    sds_1_1 = sample('sds_1_1', lower_constrained_improper_uniform(0, shape=[])
        )
    Intercept_sigma = sample('Intercept_sigma', improper_uniform(shape=[]))
    bs_sigma = sample('bs_sigma', improper_uniform(shape=[Ks_sigma]))
    zs_sigma_1_1 = sample('zs_sigma_1_1', improper_uniform(shape=[
        knots_sigma_1]))
    sds_sigma_1_1 = sample('sds_sigma_1_1',
        lower_constrained_improper_uniform(0, shape=[]))
    s_1_1 = sds_1_1 * zs_1_1
    s_sigma_1_1 = sds_sigma_1_1 * zs_sigma_1_1
    mu = Intercept + rep_vector_int_int(0, N) + matmul(Xs, bs) + matmul(Zs_1_1,
        s_1_1)
    sigma = Intercept_sigma + rep_vector_int_int(0, N) + matmul(Xs_sigma,
        bs_sigma) + matmul(Zs_sigma_1_1, s_sigma_1_1)

    @jit
    def _fori__1(n, _acc__2):
        sigma = _acc__2
        sigma = sigma.at[n - 1].set(exp_real(sigma[n - 1]))
        return sigma
    sigma = lax_fori_loop(1, N + 1, _fori__1, sigma)
    factor('_expr__3', student_t_lpdf(Intercept, 3, -13, 36))
    factor('_expr__4', normal_lpdf(zs_1_1, 0, 1))
    factor('_expr__5', student_t_lpdf(sds_1_1, 3, 0, 36) - 1 *
        student_t_lccdf(0, 3, 0, 36))
    factor('_expr__6', student_t_lpdf(Intercept_sigma, 3, 0, 10))
    factor('_expr__7', normal_lpdf(zs_sigma_1_1, 0, 1))
    factor('_expr__8', student_t_lpdf(sds_sigma_1_1, 3, 0, 36) - 1 *
        student_t_lccdf(0, 3, 0, 36))

    def _then__9(_acc__10):
        factor('_expr__12', normal_lpdf(Y, mu, sigma))
        return None

    def _else__11(acc):
        return acc
    _ = numpyro_cond(not prior_only, _then__9, _else__11, None)


def generated_quantities(*, N, Y, Ks, Xs, knots_1, Zs_1_1, Ks_sigma,
    Xs_sigma, knots_sigma_1, Zs_sigma_1_1, prior_only, Intercept, bs,
    zs_1_1, sds_1_1, Intercept_sigma, bs_sigma, zs_sigma_1_1, sds_sigma_1_1):
    s_1_1 = sds_1_1 * zs_1_1
    s_sigma_1_1 = sds_sigma_1_1 * zs_sigma_1_1
    b_Intercept = Intercept
    b_sigma_Intercept = Intercept_sigma
    return {'s_1_1': s_1_1, 's_sigma_1_1': s_sigma_1_1, 'b_Intercept':
        b_Intercept, 'b_sigma_Intercept': b_sigma_Intercept}


def map_generated_quantities(_samples, *, N, Y, Ks, Xs, knots_1, Zs_1_1,
    Ks_sigma, Xs_sigma, knots_sigma_1, Zs_sigma_1_1, prior_only):

    def _generated_quantities(Intercept, bs, zs_1_1, sds_1_1,
        Intercept_sigma, bs_sigma, zs_sigma_1_1, sds_sigma_1_1):
        return generated_quantities(N=N, Y=Y, Ks=Ks, Xs=Xs, knots_1=knots_1,
            Zs_1_1=Zs_1_1, Ks_sigma=Ks_sigma, Xs_sigma=Xs_sigma,
            knots_sigma_1=knots_sigma_1, Zs_sigma_1_1=Zs_sigma_1_1,
            prior_only=prior_only, Intercept=Intercept, bs=bs, zs_1_1=
            zs_1_1, sds_1_1=sds_1_1, Intercept_sigma=Intercept_sigma,
            bs_sigma=bs_sigma, zs_sigma_1_1=zs_sigma_1_1, sds_sigma_1_1=
            sds_sigma_1_1)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['Intercept'], _samples['bs'], _samples['zs_1_1'],
        _samples['sds_1_1'], _samples['Intercept_sigma'], _samples[
        'bs_sigma'], _samples['zs_sigma_1_1'], _samples['sds_sigma_1_1'])


def parameters_info(*, N, Y, Ks, Xs, knots_1, Zs_1_1, Ks_sigma, Xs_sigma,
    knots_sigma_1, Zs_sigma_1_1, prior_only):
    return {'Intercept': {'shape': []}, 'bs': {'shape': [Ks]}, 'zs_1_1': {
        'shape': [knots_1]}, 'sds_1_1': {'shape': []}, 'Intercept_sigma': {
        'shape': []}, 'bs_sigma': {'shape': [Ks_sigma]}, 'zs_sigma_1_1': {
        'shape': [knots_sigma_1]}, 'sds_sigma_1_1': {'shape': []}}
