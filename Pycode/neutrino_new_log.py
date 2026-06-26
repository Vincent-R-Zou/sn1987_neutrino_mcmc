import definition as df
import numpy as np
from scipy import integrate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
from numpy.polynomial.legendre import leggauss
import multiprocessing as mp
from multiprocessing import Pool
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

Distance_SN1987 = df.Distance_SN1987        # Distance of SN 1987A in unit cm
coefficient = 3.89379e-10      # Unit transformation from eV^-2 to cm^2
m_nu = df.m_nu       # neutrino mass in eV
T_nu = df.T_nu       # neutrino temperature in eV

def gamma_phi(log_m_phi, lambda_chi = 1):

    m_phi = 10**log_m_phi

    return lambda_chi**2 * m_phi / (32*np.pi)


def cross_section(E_nu, log_m_phi, log_lambda_nu, c, E_1):      # In ev^-2

    m_phi = 10**log_m_phi

    lambda_nu = 10**log_lambda_nu

    part1 = lambda_nu ** 2 /  m_phi**2       # lambda in 1e-8 unit， m_phi in eV

    beta = np.sqrt(1- m_nu**2/E_1**2)


    part2 = (1-beta*c)**2 / np.sqrt(1+beta**2- 2*beta*c)

    gamma = gamma_phi(log_m_phi)

    numerator = (2*E_1*E_nu*1e6 / m_phi**2) * (gamma / m_phi)

    denominator = (gamma / m_phi)**2 + ((2*E_1*E_nu*1e6 / m_phi**2)*(1-beta*c) -1)**2

    part3 = numerator / denominator

    ret = part1 * part2 * part3

    return ret




def d_phi(E_nu, log_m_phi, log_lambda_nu):

    E_space = np.linspace(4, df.E_max_2d, 15)

    m_phi = 10**log_m_phi

    E_1_min = np.clip((m_phi**2 - m_phi*gamma_phi(log_m_phi)) / (4*E_space*1e6), m_nu, 0.002)

    # print("E_1_min:", E_1_min)

    f_FD = lambda E_1:  1/(1+df.safe_exp(E_1/T_nu))

    integ_part = lambda c, E_1: cross_section(E_space, log_m_phi, log_lambda_nu, c, E_1) * f_FD(E_1) * E_1**2/ (4*np.pi**2)

    ratio = -df.gl2_integrate_vector(integ_part, [(-1, 1), (E_1_min, 0.002)], n=200)

    ret = IUS(E_space, ratio)(E_nu)* Distance_SN1987 * np.sqrt(1/coefficient)

    return np.clip(ret, None, 0)



def d_phi_test(E_nu, log_m_phi, log_lambda_nu):

    m_phi = 10**log_m_phi

    E_1_min = np.clip((m_phi**2 - m_phi*gamma_phi(log_m_phi)) / (4*E_nu*1e6), m_nu, 0.002)

    # print("E_1_min:", E_1_min)

    f_FD = lambda E_1:  1/(1+df.safe_exp(E_1/T_nu))

    integ_part = lambda c, E_1: cross_section(E_nu, log_m_phi, log_lambda_nu, c, E_1) * f_FD(E_1) * E_1**2/ (4*np.pi**2)

    ratio1 = lambda E_1: integrate.quad_vec(integ_part, -1, 1-1e-10, args=E_1)[0]

    ratio2 = integrate.quad_vec(ratio1, E_1_min, 0.002)[0]

    return ratio2* Distance_SN1987 * np.sqrt(1/coefficient)


d_phi_test_vec = np.vectorize(d_phi_test, excluded=['E_nu'])


if __name__ == '__main__':

    # print(cross_section(10, 100, 100, 0.5, 0.000001))
    
    E_nu = 7
    test_m_phi = 2
    test_lambda_nu = -5

    print(d_phi(np.array([E_nu]), test_m_phi, test_lambda_nu))
    print(np.exp(d_phi(np.array([E_nu]), test_m_phi, test_lambda_nu)))
    # E_space = np.linspace(4, 50, 1000)
    # plt.plot(E_space, d_phi(E_space, 2, -6))
    # plt.xlabel("E_nu (MeV)")
    # plt.ylabel("d_phi")
    # plt.title("d_phi vs E_nu for m_phi=100 MeV and lambda_nu=1e-6")
    # plt.show()

        
    # m_space = np.linspace(-3, 3, 100)

    # lambda_space = np.linspace(-9, -4, 100)

    # X, Y = np.meshgrid(m_space, lambda_space)

    # ctx = mp.get_context("spawn")
    # nproc = ctx.cpu_count()

    # tasks = [(30, mx, ly) for mx, ly in zip(X.ravel(), Y.ravel())]

    # with Pool(processes=nproc) as pool:
    #     Z = np.clip(np.array(pool.starmap(d_phi_test_vec, tasks)).reshape(X.shape), 0, 1)

    # fig = plt.figure(figsize=(10, 7))
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    # ax.set_title('3D Plot of d_phi(E_nu=30, m_phi, lambda_nu)')
    # ax.set_xlabel('m_phi (MeV)')
    # ax.set_ylabel('lambda_nu')
    # ax.set_zlabel('d_phi')

    # plt.savefig('F:/Neutrino_SI/Plots/d_phi_3d_plot_Christina_01_log.png')
    # plt.show()

