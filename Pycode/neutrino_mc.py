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
from scipy.interpolate import InterpolatedUnivariateSpline as IUS


coefficient = 3.89379e-10      # Unit transformation from eV^-2 to cm^2
Distance_SN1987 = df.Distance_SN1987        # Distance of SN 1987A in unit cm
T_nu = 2.725 * 0.7138 * 8.617e-5        # in unit ev

# gamma_phi devided by m_phi
def omega(lambda_chi = 1):

    return lambda_chi**2/ (32*np.pi)


def gamma_phi(m_phi, lambda_chi = 1):

    return lambda_chi**2 * m_phi / (32*np.pi)

def alpha(E_nu, m_phi):
    return 2*E_nu*1e6 * T_nu / m_phi**2

def alpha_m_phi(E_nu):
    return 2*E_nu*1e6 * T_nu


# Gauss legendre integration makes it wiggle heavily here
def cross_section_gl(E_nu, m_phi, m_nu  = 0):      # In ev^-2  

    x_0 = m_nu / T_nu

    beta = lambda x: np.sqrt(1- x_0**2/x**2)

    part1 = lambda x: x**2 / (df.safe_exp(x) + 1 )    # important in x \in [0., 20.]

    part2 = lambda c, x: (1-beta(x)*c)**2 / np.sqrt(1+beta(x)**2- 2*beta(x)*c)

    part3 = lambda c, x, E_nu: omega()*alpha_m_phi(E_nu)*x / (omega()**2 * m_phi**4 + (alpha_m_phi(E_nu)*x*(1-beta(x)*c) -m_phi**2)**2)

    integ_part = lambda c, x, E_nu: part1(x) * part2(c, x) * part3(c, x, E_nu)

    ret = df.gl2_integrate_vec(integ_part, [(-1, 1), (0.0, 20.)], E_nu, n=256)

    return ret


def d_phi_gl(E_nu, m_phi, lambda_nu):

    E_space = np.linspace(4, 100, 20)

    part1 = T_nu**3 * (lambda_nu*1e-8)**2 / (4*np.pi**2)

    ratio =  -part1*cross_section_gl(E_space, m_phi)* Distance_SN1987 * np.sqrt(1/coefficient)

    ret = IUS(E_space, ratio)(E_nu)

    return np.clip(ret, -1, 0)


def cross_section_mc(E_nu, m_phi, m_nu = 0):
    # ... define your part1, part2, part3 here ...
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
    integ(batch_integrand, nitn=10, neval=1000)
    
    # Step 2: Calculate the final high-precision value
    result = integ(batch_integrand, nitn=10, neval=1000)
    
    return result.mean


def d_phi_mc(E_nu, m_phi, lambda_nu):

    E_space = np.linspace(4, 100, 1000)

    cross_section_vec = np.vectorize(cross_section_mc, excluded=['m_phi', 'm_nu'])

    part1 = T_nu**3 * (lambda_nu*1e-8)**2 / (4*np.pi**2)

    ratio =  -part1*cross_section_vec(E_space, m_phi)* Distance_SN1987 * np.sqrt(1/coefficient)

    ret = df.func_build(E_nu, E_space, ratio)

    return np.clip(ret, -1, 0)



if __name__ == '__main__':

    E_space = np.linspace(4, 100, 1000)

    # print(d_phi_gl(np.array([30, 40]), 100, 100))

    # plt.plot(E_space, d_phi_gl(E_space, 100, 100))
    # plt.xlabel(r"$E_\nu$ (MeV)")
    # plt.ylabel(r"$d\phi$")
    # plt.title(r"$d\phi$ vs $E_\nu$ for $m_\phi$=100 MeV and $\lambda_\nu$=1e-6")
    # plt.show()

    m_space = np.linspace(0, 2000, 100)

    lambda_space = np.linspace(0, 2000, 100)

    X, Y = np.meshgrid(m_space, lambda_space)

    ctx = mp.get_context("spawn")
    nproc = ctx.cpu_count()

    tasks = [(30, mx, ly) for mx, ly in zip(X.ravel(), Y.ravel())]

    with Pool(processes=nproc) as pool:
        Z = -np.array(pool.starmap(d_phi_gl, tasks)).reshape(X.shape)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(r'3D Plot of $\frac{d\phi}{\phi}(E_\nu=30 MeV, m_\phi, \lambda_\nu)$ [Christina’s version]')
    ax.set_xlabel(r'$m_\phi$ (eV)')
    ax.set_ylabel(r'$\lambda_\nu (10^{-8})$')
    ax.set_zlabel(r'$\frac{d\phi}{\phi}$')

    # plt.savefig("F:/Neutrino_SI/Plots/d_phi_3d_plot_Christina_01.png")

    plt.show()