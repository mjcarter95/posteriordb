from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import exp_real, max_vector, min_vector, rep_vector_int_int


def growth_factor_weibull(t, omega, theta):
    return 1 - exp_real(-true_divide(t, theta) ** omega)


def growth_factor_loglogistic(t, omega, theta):
    pow_t_omega = t ** omega
    return true_divide(pow_t_omega, pow_t_omega + theta ** omega)


def convert_inputs(inputs):
    growthmodel_id = inputs['growthmodel_id']
    n_data = inputs['n_data']
    cohort_id = array(inputs['cohort_id'], dtype=dtype_long)
    t_idx = array(inputs['t_idx'], dtype=dtype_long)
    n_cohort = inputs['n_cohort']
    cohort_maxtime = array(inputs['cohort_maxtime'], dtype=dtype_long)
    n_time = inputs['n_time']
    t_value = array(inputs['t_value'], dtype=dtype_float)
    premium = array(inputs['premium'], dtype=dtype_float)
    loss = array(inputs['loss'], dtype=dtype_float)
    return {'growthmodel_id': growthmodel_id, 'n_data': n_data, 'cohort_id':
        cohort_id, 't_idx': t_idx, 'n_cohort': n_cohort, 'cohort_maxtime':
        cohort_maxtime, 'n_time': n_time, 't_value': t_value, 'premium':
        premium, 'loss': loss}


def model(*, growthmodel_id, n_data, cohort_id, t_idx, n_cohort,
    cohort_maxtime, n_time, t_value, premium, loss):
    omega = sample('omega', lower_constrained_improper_uniform(0, shape=[]))
    theta = sample('theta', lower_constrained_improper_uniform(0, shape=[]))
    LR = sample('LR', lower_constrained_improper_uniform(0, shape=[n_cohort]))
    mu_LR = sample('mu_LR', improper_uniform(shape=[]))
    sd_LR = sample('sd_LR', lower_constrained_improper_uniform(0, shape=[]))
    loss_sd = sample('loss_sd', lower_constrained_improper_uniform(0, shape=[])
        )
    gf = empty([n_time], dtype=dtype_float)

    @jit
    def _fori__1(i, _acc__2):
        gf = _acc__2
        gf = gf.at[i - 1].set(growth_factor_weibull(t_value[i - 1], omega,
            theta) if growthmodel_id == 1 else growth_factor_loglogistic(
            t_value[i - 1], omega, theta))
        return gf
    gf = lax_fori_loop(1, n_time + 1, _fori__1, gf)
    lm = empty([n_data], dtype=dtype_float)

    @jit
    def _fori__3(i, _acc__4):
        lm = _acc__4
        lm = lm.at[i - 1].set(LR[cohort_id[i - 1] - 1] * premium[cohort_id[
            i - 1] - 1] * gf[t_idx[i - 1] - 1])
        return lm
    lm = lax_fori_loop(1, n_data + 1, _fori__3, lm)
    observe('_mu_LR__5', normal(0, array(0.5, dtype=dtype_float)), mu_LR)
    observe('_sd_LR__6', lognormal(0, array(0.5, dtype=dtype_float)), sd_LR)
    observe('_LR__7', lognormal(mu_LR, sd_LR), LR)
    observe('_loss_sd__8', lognormal(0, array(0.7, dtype=dtype_float)), loss_sd
        )
    observe('_omega__9', lognormal(0, array(0.5, dtype=dtype_float)), omega)
    observe('_theta__10', lognormal(0, array(0.5, dtype=dtype_float)), theta)
    observe('_loss__11', normal(lm, (loss_sd * premium)[cohort_id - 1]), loss)


def generated_quantities(*, growthmodel_id, n_data, cohort_id, t_idx,
    n_cohort, cohort_maxtime, n_time, t_value, premium, loss, omega, theta,
    LR, mu_LR, sd_LR, loss_sd):
    gf = empty([n_time], dtype=dtype_float)

    @jit
    def _fori__12(i, _acc__13):
        gf = _acc__13
        gf = gf.at[i - 1].set(growth_factor_weibull(t_value[i - 1], omega,
            theta) if growthmodel_id == 1 else growth_factor_loglogistic(
            t_value[i - 1], omega, theta))
        return gf
    gf = lax_fori_loop(1, n_time + 1, _fori__12, gf)
    lm = empty([n_data], dtype=dtype_float)

    @jit
    def _fori__14(i, _acc__15):
        lm = _acc__15
        lm = lm.at[i - 1].set(LR[cohort_id[i - 1] - 1] * premium[cohort_id[
            i - 1] - 1] * gf[t_idx[i - 1] - 1])
        return lm
    lm = lax_fori_loop(1, n_data + 1, _fori__14, lm)
    loss_sample = empty([n_cohort, n_time], dtype=dtype_float)
    step_ratio = empty([n_cohort, n_time], dtype=dtype_float)

    @jit
    def _fori__16(i, _acc__17):
        loss_sample, step_ratio = _acc__17
        step_ratio = step_ratio.at[i - 1].set(rep_vector_int_int(1, n_time))
        loss_sample = loss_sample.at[i - 1].set(LR[i - 1] * premium[i - 1] * gf
            )
        return loss_sample, step_ratio
    loss_sample, step_ratio = lax_fori_loop(1, n_cohort + 1, _fori__16, (
        loss_sample, step_ratio))
    mu_LR_exp = exp_real(mu_LR)
    loss_prediction = empty([n_cohort, n_time], dtype=dtype_float)

    @jit
    def _fori__18(i, _acc__19):
        loss_prediction = _acc__19
        loss_prediction = loss_prediction.at[cohort_id[i - 1] - 1, t_idx[i -
            1] - 1].set(loss[i - 1])
        return loss_prediction
    loss_prediction = lax_fori_loop(1, n_data + 1, _fori__18, loss_prediction)

    @jit
    def _fori__20(i, _acc__21):
        step_ratio = _acc__21

        @jit
        def _fori__22(j, _acc__23):
            step_ratio = _acc__23
            step_ratio = step_ratio.at[i - 1, j - 1].set(true_divide(gf[
                t_idx[j - 1] - 1], gf[t_idx[j - 1 - 1] - 1]))
            return step_ratio
        step_ratio = lax_fori_loop(2, n_time + 1, _fori__22, step_ratio)
        return step_ratio
    step_ratio = lax_fori_loop(1, n_cohort + 1, _fori__20, step_ratio)

    @jit
    def _fori__24(i, _acc__25):
        loss_prediction = _acc__25

        @jit
        def _fori__26(j, _acc__27):
            loss_prediction = _acc__27
            loss_prediction = loss_prediction.at[i - 1, j - 1].set(
                loss_prediction[i - 1, j - 1 - 1] * step_ratio[i - 1, j - 1])
            return loss_prediction
        loss_prediction = lax_fori_loop(cohort_maxtime[i - 1] + 1, n_time +
            1, _fori__26, loss_prediction)
        return loss_prediction
    loss_prediction = lax_fori_loop(1, n_cohort + 1, _fori__24, loss_prediction
        )
    ppc_minLR = min_vector(LR)
    ppc_maxLR = max_vector(LR)
    ppc_EFC = 0

    @jit
    def _fori__28(i, _acc__29):
        ppc_EFC = _acc__29
        ppc_EFC = ppc_EFC + loss_prediction[i - 1, n_time - 1
            ] - loss_prediction[i - 1, cohort_maxtime[i - 1] - 1]
        return ppc_EFC
    ppc_EFC = lax_fori_loop(1, n_cohort + 1, _fori__28, ppc_EFC)
    return {'gf': gf, 'lm': lm, 'loss_sample': loss_sample, 'step_ratio':
        step_ratio, 'mu_LR_exp': mu_LR_exp, 'loss_prediction':
        loss_prediction, 'ppc_minLR': ppc_minLR, 'ppc_maxLR': ppc_maxLR,
        'ppc_EFC': ppc_EFC}


def map_generated_quantities(_samples, *, growthmodel_id, n_data, cohort_id,
    t_idx, n_cohort, cohort_maxtime, n_time, t_value, premium, loss):

    def _generated_quantities(omega, theta, LR, mu_LR, sd_LR, loss_sd):
        return generated_quantities(growthmodel_id=growthmodel_id, n_data=
            n_data, cohort_id=cohort_id, t_idx=t_idx, n_cohort=n_cohort,
            cohort_maxtime=cohort_maxtime, n_time=n_time, t_value=t_value,
            premium=premium, loss=loss, omega=omega, theta=theta, LR=LR,
            mu_LR=mu_LR, sd_LR=sd_LR, loss_sd=loss_sd)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['omega'], _samples['theta'], _samples['LR'],
        _samples['mu_LR'], _samples['sd_LR'], _samples['loss_sd'])


def parameters_info(*, growthmodel_id, n_data, cohort_id, t_idx, n_cohort,
    cohort_maxtime, n_time, t_value, premium, loss):
    return {'omega': {'shape': []}, 'theta': {'shape': []}, 'LR': {'shape':
        [n_cohort]}, 'mu_LR': {'shape': []}, 'sd_LR': {'shape': []},
        'loss_sd': {'shape': []}}
