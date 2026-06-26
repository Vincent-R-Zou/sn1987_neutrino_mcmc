import definition as df
import numpy as np
from scipy import integrate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
from numpy.polynomial.legendre import leggauss
import multiprocessing as mp
from multiprocessing import Pool
import vegas
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.signal import savgol_filter  # Import the smoothing filter

coefficient = 3.89379e-10      # Unit transformation from eV^-2 to cm^2
Distance_SN1987 = df.Distance_SN1987        # Distance of SN 1987A in unit cm
T_nu = df.T_nu
m_nu = 0

# gamma_phi devided by m_phi
def omega(lambda_chi = 1):

    return lambda_chi**2/ (32*np.pi)


def gamma_phi(log_m_phi, lambda_chi = 1):

    return lambda_chi**2 * 10**log_m_phi / (32*np.pi)


def alpha_m_phi(E_nu):
    return 2*E_nu*1e6 * T_nu


def alpha(E_nu, m_phi):
    return 2*E_nu*df.T_nu*1e6 / m_phi**2


# Gauss legendre integration makes it wiggle heavily here
def cross_section_gl(E_nu, log_m_phi, m_nu  = 0):      # In ev^-2  

    m_phi = 10**log_m_phi

    x_0 = m_nu / T_nu

    beta = lambda x: np.sqrt(1- x_0**2/x**2)

    part1 = lambda x: x**2 / (df.safe_exp(x) + 1 )    # important in x \in [0., 20.]

    part2 = lambda c, x: (1-beta(x)*c)**2 / np.sqrt(1+beta(x)**2- 2*beta(x)*c)

    part3 = lambda c, x, E_nu: omega()*alpha_m_phi(E_nu)*x / (omega()**2 * m_phi**4 + (alpha_m_phi(E_nu)*x*(1-beta(x)*c) -m_phi**2)**2)

    integ_part = lambda c, x, E_nu: part1(x) * part2(c, x) * part3(c, x, E_nu)

    ret = df.gl2_integrate_vec(integ_part, [(-1, 1), (x_0, 20.)], E_nu, n=256)

    return ret


def d_phi_gl(E_nu, log_m_phi, log_lambda_nu):
    
    E_space = np.linspace(4, 100, 1000)

    lambda_nu = 10**log_lambda_nu

    part1 = T_nu**3 * lambda_nu**2 / (4*np.pi**2)

    ratio =  -part1*cross_section_gl(E_space, log_m_phi, m_nu)* Distance_SN1987 * np.sqrt(1/coefficient)

    ret = df.func_build(E_nu, E_space, ratio)

    return np.clip(ret, -1, 0)


def cross_section_mc(E_nu, log_m_phi, m_nu = 0):
    # ... define your part1, part2, part3 here ...

    m_phi = 10**log_m_phi

    x_0 = m_nu / T_nu

    beta = lambda x: np.sqrt(1- x_0**2/x**2)

    part1 = lambda x: x**2 / (df.safe_exp(x) + 1 )    # important in x \in [0., 20.]

    part2 = lambda c, x: (1-beta(x)*c)**2 / np.sqrt(1+beta(x)**2- 2*beta(x)*c)

    part3 = lambda c, x: omega()*alpha_m_phi(E_nu)*x / (omega()**2 * m_phi**4 + (alpha_m_phi(E_nu)*x*(1-beta(x)*c) -m_phi**2)**2)

    @vegas.batchintegrand
    def batch_integrand(x_arr):
        # x_arr contains [c, x] pairs
        c = x_arr[:, 0]
        x = x_arr[:, 1]
        return part1(x) * part2(c, x) * part3(c, x)

    # Define the 2D integration space: c in [-1, 1], x in [0, 20]
    integ = vegas.Integrator([[-1, 1], [0.0, 20.0]])
    
    # Step 1: Train the grid (find the peak)
    integ(batch_integrand, nitn=10, neval=2000)
    
    # Step 2: Calculate the final high-precision value
    result = integ(batch_integrand, nitn=10, neval=5000)
    
    return result.mean


def d_phi_mc(E_nu, log_m_phi, log_lambda_nu):

    E_space = np.linspace(4, 100, 1000)

    lambda_nu = 10**log_lambda_nu

    cross_section_vec = np.vectorize(cross_section_mc, excluded=['m_phi', 'm_nu'])

    part1 = T_nu**3 * lambda_nu**2 / (4*np.pi**2)

    ratio =  -part1*cross_section_vec(E_space, log_m_phi)* Distance_SN1987 * np.sqrt(1/coefficient)

    ret = df.func_build(E_nu, E_space, ratio)  

    return np.clip(ret, -1, 0)



def normalized_rate(E_nu, m_phi, x_0):      # In ev^-2  

    # x_0 = m_nu / T_nu

    beta = lambda x: np.sqrt(1- x_0**2/x**2)

    part1 = lambda x: x**2 / (df.safe_exp(x) + 1 )    # important in x \in [0., 20.]

    part2 = lambda c, x: (1-beta(x)*c)**2 / np.sqrt(1+beta(x)**2- 2*beta(x)*c)

    part3 = lambda c, x: omega()*alpha(E_nu, m_phi)*x / (omega()**2  + (alpha(E_nu, m_phi)*x*(1-beta(x)*c) -1)**2)

    integ_part = lambda c, x: part1(x) * part2(c, x) * part3(c, x)

    ret = df.gl2_integrate(integ_part, [(-1, 1), (x_0, 20.)], n=1024)

    return ret


def compute_row(x_0, E_nu_0, m_phi):
    """
    Worker function that computes the inner loop (over m_phi_space) 
    for a single x_0 value.
    """
    return [normalized_rate(E_nu_0, m_phi, x_0) for m_phi in m_phi]


def normalized_rate_x(E_nu, m_phi, x_0):      # In ev^-2  

# x_0 = m_nu / T_nu

    beta = lambda x: np.sqrt(1- x_0**2/x**2)

    # def cpeak(x):
    #     return 1/beta(x) * (1-1/alpha(E_nu, m_phi)/x)

    part1 = lambda x: x**2 / (df.safe_exp(x) + 1 )    # important in x \in [0., 20.]

    part2 = lambda c, x: (1-beta(x)*c)**2 / np.sqrt(1+beta(x)**2- 2*beta(x)*c)

    part3 = lambda c, x, m_phi: omega()*alpha(E_nu, m_phi)*x / (omega()**2  + (alpha(E_nu, m_phi)*x*(1-beta(x)*c) -1)**2)

    integ_part = lambda c, x, m_phi: part1(x) * part2(c, x) * part3(c, x, m_phi)

    ret = df.gl2_integrate_vec(integ_part, [(-1, 1), (x_0, 20.)], m_phi, n=1024)

    return ret


def compute_row_x(x_0, E_nu_0, m_phi):
    """
    Worker function that computes the inner loop (over m_phi_space) 
    for a single x_0 value.
    """
    return normalized_rate_x(E_nu_0, m_phi, x_0)



if __name__ == '__main__':

    
    m_phi_space = np.logspace(0, 4, 50)  # Example m_phi values
    x_0_space = np.logspace(-2, 1, 50)     # Example x_0 values
    
    print(normalized_rate_x(30, m_phi_space, 0.01))
    print(compute_row_x(0.01, 30, m_phi_space))

