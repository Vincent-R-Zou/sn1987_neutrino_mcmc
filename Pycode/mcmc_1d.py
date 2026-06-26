import numpy as np
import SignalRate_1d
from scipy import integrate
import Data
import definition as df
import Errors
from scipy import optimize
import emcee
import matplotlib.pyplot as plt
import corner
import multiprocessing as mp
import xlwt
from multiprocessing.dummy import Pool as ThreadPool
from dataclasses import dataclass, field
from SignalRate_1d import log_scale



E_max = df.E_max_1d
t_max = df.t_max_1d

E_min_K = df.E_min_K
E_min_I = df.E_min_I
E_min_B = df.E_min_B

path_str = '1d_8p_1e6_p32_log'

@dataclass
class ModelSettings:

# =========================
# adjustable
# =========================

    # 1/6 parameters you want to fix: alternative for 'R_c', 'T_c', 'tau_c', 'M_a', 'T_a', 'tau_a', 'm_phi', 'lambda_nu', devided by comma.   
    fix_param: list = field(default_factory=lambda: [])

    # 2/6 percison (>=16), steps (> discards), discards (>1000) and thin (multiples of 5)
    scale_factor: float = 1      
    percision: int = 32   
    steps, dicards, thin = 50000, 30000, 20

    # 3/6 detectors: alternative for 'K'(Kamiokande), 'B'(Baksan) and 'I'(IMB), devided by comma as well.
    detector: list = field(default_factory=lambda: ['K', 'B', 'I'])

    # 4/6 initial value imported in mcmc (values should be adjusted according to BOUNDS_ALL) 13.75, 4.90, 4.74, 0.17, 2.33, 0.54
    INITIALS: dict = field(default_factory=lambda: {
        'R_c':          15.00,
        'T_c':          5.00,
        'tau_c':        5.00,
        'M_a':          1.00,
        'T_a':          2.50,
        'tau_a':        0.50,
        'm_phi':        2,
        'lambda_nu':    -6,
    })

    # log_m_phi 2, log_lambda_nu -6
    
    # 5/6 path  8p_p8_penalty
    path_figure: str = r"F:/Neutrino_SI/Figures/" + str(path_str) + ".png"
    path_chain:  str = r"F:/Neutrino_SI/Data/" + str(path_str) + ".npy"

    # 6/6 priors (values are adjustable)
    BOUNDS_ALL: dict = field(default_factory=lambda: {
        "R_c":        [(1, 100),    (0, 0)],
        "T_c":        [(1, 10),     (0, 0)],
        "tau_c":      [(1, 10),     (0, 0)],
        "M_a":        [(0.001, 2),    (0, 0)],
        "T_a":        [(1, 10),     (0, 0)],
        "tau_a":      [(0.1, 2),    (0, 0)],
        "m_phi":      [(-1, 4),    (0, 0)],
        "lambda_nu":  [(-10, -3),    (0, 0)],
    })
    # log_m_phi \in [-2, 4], log_lambda_nu \in [-10, -4]

# =========================
# adjustment of the following definitions is not recommanded
# =========================

    PARAMS_ALL: list = field(default_factory=lambda: [
        'R_c', 'T_c', 'tau_c', 'M_a',
        'T_a', 'tau_a', 'm_phi', 'lambda_nu'
    ])

    LABELS: list = field(default_factory=lambda: [
        '$R_c$', '$T_c$', r'$\tau_c$',
        '$M_a$', '$T_a$', r'$\tau_a$',
        r'log($m_{\Phi}$)/eV', r'$log({\lambda}_{\Phi \nu})$',        
        # r'$m_{\Phi}$/eV', r'${\lambda}_{\Phi \nu}/10^{-8}$'
    ])

    ranges_K: list = field(default_factory=lambda: [(0, t_max), (E_min_K, E_max),  (-1, 1)])
    ranges_I: list = field(default_factory=lambda: [(0, t_max), (E_min_I, E_max), (-1, 1)])
    ranges_B: list = field(default_factory=lambda: [(0, t_max), (E_min_B, E_max), (-1, 1)])


def _init_from_settings(cfg: ModelSettings):

    global fix_param, detector, LABELS, INITIALS
    global PARAMS_ALL, BOUNDS_ALL, indexes_in, indexes_out, PARAMS, BOUNDS
    global fixed_values, initial
    global ranges_K, ranges_I, ranges_B
    global scale_factor, percision
    global t_K, E_K, c_K, dE_K, B_K
    global t_I, E_I, c_I, dE_I, B_I
    global t_B, E_B, c_B, dE_B, B_B
    global path_figure, path_chain
    global kx, bx, ix
    global steps, discards, thin
    # global t_max, E_max

    fix_param   = cfg.fix_param
    detector     = cfg.detector
    kx, bx, ix = [1 if x in detector else 0 for x in ["K", "B", "I"]]

    LABELS      = cfg.LABELS.copy()    
    INITIALS    = cfg.INITIALS
    PARAMS_ALL  = cfg.PARAMS_ALL
    BOUNDS_ALL  = cfg.BOUNDS_ALL

    ranges_K    = cfg.ranges_K
    ranges_I    = cfg.ranges_I
    ranges_B    = cfg.ranges_B

    scale_factor = cfg.scale_factor
    percision    = cfg.percision
    steps, discards, thin = cfg.steps, cfg.dicards, cfg.thin
    # t_max, E_max = cfg.t_max, cfg.E_max

    path_figure = cfg.path_figure
    path_chain  = cfg.path_chain

    indexes_in = [PARAMS_ALL.index(name) for name in fix_param]

    indexes_out = [PARAMS_ALL.index(name) for name in PARAMS_ALL if name not in fix_param]

    PARAMS = [x for x in PARAMS_ALL if x not in fix_param]

    BOUNDS = {key: value for key, value in BOUNDS_ALL.items() if key not in fix_param}

    fixed_values = [INITIALS[p] for p in fix_param]

    initial = [INITIALS[p] for p in PARAMS]

    for idx, val in sorted(zip(indexes_in, fixed_values), reverse=True):
        LABELS.pop(idx)

    (t_K, E_K, c_K, dE_K, B_K,
     t_I, E_I, c_I, dE_I, B_I,
     t_B, E_B, c_B, dE_B, B_B) = (
        Data.Kam_t_valid,     Data.Kam_E_valid,     Data.Kam_c_valid,     Data.Kam_dE_valid,     Data.Kam_B_valid_2009,
        Data.IMB_t_valid,     Data.IMB_E_valid,     Data.IMB_c_valid,     Data.IMB_dE_valid,     Data.IMB_B_valid,
        Data.Baksan_t_valid,  Data.Baksan_E_valid,  Data.Baksan_c_valid,  Data.Baksan_dE_valid,  Data.Baksan_B_valid
    )


settings = ModelSettings()
_init_from_settings(settings)


def insert_args(theta):
    theta_new = list(theta.copy())
    args_list = [0]*len(PARAMS_ALL)
    for idx, val in sorted(zip(indexes_in, fixed_values), reverse=False):
        args_list[idx] = val
    for idx, val in sorted(zip(indexes_out, theta_new), reverse=False):
        args_list[idx] = val
    return args_list


def log_likelihood_K(theta):

    if kx:

        args_list = insert_args(theta)

        # print("args_list in log_likelihood_K:", args_list)

        fun1_K = lambda x: SignalRate_1d.SR_K(t_K, x, c_K, *args_list) * Errors.Error_E(x, E_K, dE_K)

        part1_K = df.gl3_integrate(SignalRate_1d.SR_K, ranges_K, percision, args=args_list)

        # part2_K = integrate.quad_vec(fun1_K, E_min_K, data_limit()[1])[0]

        part2_K = np.clip(df.gl_integrate(fun1_K, E_min_K, E_max), 1e-100, None)

        part3_K = B_K / 2

        ret_K = -part1_K + np.sum(np.log(part3_K + part2_K))

        return ret_K * scale_factor

    return 0

# print(initial)
# print('test log_likelihood with initial:', log_likelihood_K(initial)/1e3)

# IMB
def log_likelihood_I(theta):

    if ix:

        args_list = insert_args(theta)

        fun1_I = lambda x: SignalRate_1d.SR_I(t_I, x, c_I, *args_list)*Errors.Error_E(x, E_I, dE_I)

        fun2_I = lambda x, c, t: SignalRate_1d.SR_I(t, x, c, *args_list)*Errors.Error_E(x, E_I, dE_I)

        part1_I = df.gl3_integrate(SignalRate_1d.SR_I, ranges_I, percision, args=args_list)

        # part2_I = np.clip(integrate.quad_vec(fun1_I, E_min_I , data_limit()[1])[0], 1e-100, None)

        part2_I = np.clip(df.gl_integrate(fun1_I, E_min_I , E_max), 1e-100, None)

        part3_I = B_I/2

        part4_I= df.gl2_integrate_vec(fun2_I, ((E_min_I, E_max), (-1,1)), t_I, percision)

        ret_I = -0.9055*part1_I  + np.sum(0.035*part4_I) + np.sum(np.log(part3_I + part2_I))
    
        return ret_I*scale_factor
    
    return 0


# Baksan
def log_likelihood_B(theta):

    if bx:

        args_list = insert_args(theta)

        fun1_B = lambda x: SignalRate_1d.SR_B(t_B, x, c_B, *args_list)*Errors.Error_E(x, E_B, dE_B)

        part1_B = df.gl3_integrate(SignalRate_1d.SR_B, ranges_B, percision, args=args_list)

        # part2_B = integrate.quad_vec(fun1_B, E_min_B, data_limit()[1])[0]

        part2_B = np.clip(df.gl_integrate(fun1_B, E_min_B, E_max), 1e-100, None)

        part3_B = B_B/2

        ret_B = - part1_B + np.sum(np.log(part3_B + part2_B))

        return ret_B*scale_factor
    
    return 0


# Total
def log_likelihood(theta):

    ret = log_likelihood_K(theta) +log_likelihood_I(theta) +log_likelihood_B(theta)

    return ret


def log_prior(theta):
    p = dict(zip(PARAMS, theta))
    for name, [(lo, hi), (plo, phi)] in BOUNDS.items():
        if not lo <= p[name] <= hi:
            return -np.inf
    return 0.0


def log_penalty(theta):
    p = dict(zip(PARAMS, theta))
    chi2 = 0
    for name, [(lo, hi), (plo, phi)] in BOUNDS.items():
        sigma = (phi-plo)/10
        if p[name]<=plo and p[name]>lo:
            chi2 += ((p[name]-plo)/sigma)**2
        elif p[name]>=phi and p[name]<hi:
            chi2 += ((p[name]-phi)/sigma)**2
        elif p[name]>hi or p[name]<lo:
            return -np.inf
        else:
            chi2 += 0
    return - chi2*scale_factor/2


def log_prob(theta):

    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)



def save_mcmc_result(flat_samples, path):
    my_workbook = xlwt.Workbook()
    sheet = my_workbook.add_sheet('mcmc_result')
    for i in range(flat_samples.shape[0]):
        for j in range(flat_samples.shape[1]):
            sheet.write(i, j, flat_samples[i][j])
    my_workbook.save(path)



if __name__ == '__main__':

    print("... information check ...")

    print('parameters to be fitted:', PARAMS)

    # print('fixed parameters:', fix_param if len(fix_param) != 0 else 'Null')

    print('data from detector(s):', detector)

    print('percision:', percision)

    print('saving path:',  path_str)

    print('initial:', initial)

    print('test log_likelihood with initial:', log_likelihood(initial))

    print('log_scale:', log_scale)

    # print('filename for spectrum:', filename)

    pos = initial + 1e-4 * np.random.randn(32, len(initial))   

    nwalkers, ndim = pos.shape

    ctx = mp.get_context("spawn")
    nproc = min(ctx.cpu_count(), nwalkers)
    with ctx.Pool(processes=nproc) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob, pool=pool
            )
        sampler.run_mcmc(pos, steps, progress=True)

    flat_samples = sampler.get_chain(discard = discards, flat=True
                                     , thin=thin
                                     )

    print("auto-correlation time:", sampler.get_autocorr_time(tol=0))
    #scale = 1, p=32: [710.12804449 573.46509797 443.89034733 511.33832749 522.06222499 478.93743603 770.46768203 534.12874907]
    #scale = 1000, p=32: [1090.40074782  396.51540139  918.60191952  749.93327084  481.33658351 322.65577749  338.77554105 1088.43402439]

    print(flat_samples.shape)

    # save_mcmc_result(flat_samples, path= path_chain)
    np.save(path_chain, flat_samples)

    # print("best fit values: ")
    print("best fit:")
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc) 
        # print(labels[i] + "={0:.3f}".format(mcmc[1],mcmc[1]-q[0],mcmc[1]+q[1]))
        print(LABELS[i] + "={0:.3f}".format(mcmc[1]))

    if not len(initial)-1:      # plot histgram for one dimentional parameter
        plt.hist(flat_samples[:, 0], 100, color="k", histtype="step")
        plt.xlabel(r"$\theta_1$")
        plt.ylabel(r"$p(\theta_1)$")
        plt.gca().set_yticks([])
    else:       # plot corner for more than one dimentional parameter
        fig = corner.corner(
            flat_samples, quantiles=[0.16, 0.5, 0.84],show_titles=True, title_kwargs={"fontsize": 12}, smooth = 1, 
            labels=LABELS, 
        )

    plt.savefig(path_figure)
    plt.show()