import definition as df
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import neutrino_new
import neutrino_new_log
from Simulation_Spectrum import time_limit
from Simulation_Spectrum import flux_emulator_e, flux_emulator_x
import Data
import Errors

m_e = df.m_e      
m_p = df.m_p       
m_n = df.m_n       
Planck_h = df.Planck_h    
LightSpeed = df.LightSpeed       
Distance_SN1987 = df.Distance_SN1987/1e5        # Distance of SN 1987A in unit km
delta_m = df.delta_m 
Delta_m = m_n - m_p     
f_d_IMB = df.f_d_IMB
tau_d_IMB = df.tau_d_IMB
m_sun = df.m_sun       

N_K = df.N_K
N_I = df.N_I
N_B = df.N_B
Ue1_2 = df.Ue1_2
log_scale = True

# The number of targets (=free protons) in the detectors (extracted from TABLE VI of arXiv:2308.01403v2)
def N_p(detector): 

    if detector == 'K':
        return 1.43e32
    elif detector == 'I':
        return 4.55e32
    elif detector == 'B':
        return 1.87e31


# Electron momentum in natural unit
def p_e(E_e):
    sqrt_arg = np.clip(E_e**2 - m_e**2, 0, None)
    result = np.sqrt(sqrt_arg)
    return result


# cross section at cosθ (based on arXiv:astro-ph/0302055v2 29 Apr 2003)
def cs_at_c(c, E_nu):

    G_F = 1.16637e-11       # Fermi coupling extracted from µ-decay
    cc = 0.9746     # cosine of the Cabibbo angle
    xi = 3.706      # ξ = κp − κn = 3.706
    g_10 = -1.27
    M = (m_p+m_n)/2
    MV2 = 0.71e6
    MA2 = 1e6
    epsilon = E_nu/m_p
    k = (1+epsilon)**2-(epsilon*c)**2       # A parameter in E_e

    # Electron energy at \nu
    E_e = ((E_nu - delta_m)*(1+epsilon) + epsilon*c*np.sqrt((E_nu-delta_m)**2-m_e**2*k))/k

    # parameters in well-known |M2
    t = m_n**2-m_p**2 - 2*m_p*(E_nu - E_e)
    f1 = (1 - (1+xi)*t/(4*M**2))/((1-t/(4*M**2))*(1-t/MV2)**2)
    f2 = xi/((1-t/(4*M**2))*(1-t/MV2)**2)
    g1 = g_10/(1-t/MA2)**2
    A = M**2*(f1**2-g1**2)*(t-m_e**2) - M**2*Delta_m**2*(f1**2+g1**2)-2*m_e**2*M*Delta_m*g1*(f1+f2)
    B = t*g1*(f1+f2)
    C = (f1**2+g1**2)/4
    s_u = 2*m_p*(E_nu+E_e)-m_e**2
    s_mp2 = 2*m_p*E_nu

    M2 = A - s_u*B + s_u**2*C

    # cross section at time t
    cs_at_t = G_F**2 * cc**2 * M2/ (2*np.pi*s_mp2**2)

    # cross section at E_e
    cs_at_Ee = 2 * m_p * cs_at_t

    alpha = 1/137.036

    # radiative corrections
    radiative_c = 1 + alpha/np.pi * (6+1.5*np.log(m_p/(2*E_e))+1.2*(m_e/E_e)**1.5)

    correction = 3.89379e-22        # Unit transformation from MeV^2 to cm^2

    result = p_e(E_e) * epsilon * cs_at_Ee / (1 + epsilon*(1-c*E_e/p_e(E_e)))

    return correction * radiative_c * result        # unit cm^2


# Considering cooling phase only
def Flux_Cooling(t, E_nu, *args):

    R_c, T_c, tau_c, m_phi, lambda_nu = args

    T_ct = T_c*np.exp(-t/4/tau_c)

    g_nu = E_nu**2/(1+np.exp(E_nu/T_ct)) 

    flux = 1/(4*np.pi*Distance_SN1987**2) * np.pi * LightSpeed/ (Planck_h*LightSpeed)**3 * (4*np.pi * R_c**2 * g_nu)
    
    return flux + neutrino_new.d_phi(E_nu, m_phi, lambda_nu) * flux


# Total Flux
def Flux_total_e(t, E_nu, *args):

    R_c, T_c, tau_c, M_a, T_a, tau_a, m_phi, lambda_nu= args   # In units of km, MeV, s respectively

    j_kt = lambda t: np.exp(-(t/tau_a)**2)
    

    # accretion

    m, Y_n = 2, 0.6

    cs = 4.8e-44*E_nu**2 / (1+E_nu/260)     

    limit = np.clip(t/tau_a, 0, 1)

    T_at = T_a + (0.6*T_c- T_a)* limit**m    
    
    m_n_SI = 1.6749275e-27  # kg

    N_nt = Y_n/m_n_SI * M_a * m_sun * (T_a/T_at)**6 * j_kt(t)/(1+2*t)       #unit 1

    E_e = (E_nu-Delta_m)/(1-E_nu/m_n)

    g_e = E_e**2/(1+np.exp(E_e/T_at)) 

    flux_accretion = 1/(4*np.pi*Distance_SN1987**2 * 1e10) * 8*np.pi * LightSpeed/ (Planck_h*LightSpeed)**3 * N_nt * cs * g_e


    # cooling with time-shift

    t_cool = t-tau_a

    T_ct = T_c*np.exp(-t_cool/4/tau_c)

    g_nu = E_nu**2/(1+df.safe_exp(E_nu/T_ct))
    
    flux_cooling = 1/(4*np.pi*Distance_SN1987**2) * np.pi * LightSpeed/ (Planck_h*LightSpeed)**3 * (4*np.pi * R_c**2 * g_nu)

    flux = flux_accretion + (1 - j_kt(t)) *flux_cooling

    if log_scale:
        return flux * np.exp(neutrino_new_log.d_phi(E_nu, m_phi, lambda_nu))
    else:
        return flux * np.exp(neutrino_new.d_phi(E_nu, m_phi, lambda_nu))      # in unit cm^-2 s^-1 MeV^-1


def Flux_total_x(t, E_nu, *args):   
    # accretion phase doesn't take effect on heavier flavor neutrinos

    R_c, T_c, tau_c, M_a, T_a, tau_a, m_phi, lambda_nu= args   # In units of km, MeV, s respectively

    j_kt = lambda t: np.exp(-(t/tau_a)**2)

    # cooling with time-shift

    t_cool = t-tau_a

    T_ct = 1.2*T_c*np.exp(-t_cool/4/tau_c)

    g_nu = E_nu**2/(1+df.safe_exp(E_nu/T_ct))
    
    flux_cooling = 1/(4*np.pi*Distance_SN1987**2) * np.pi * LightSpeed/ (Planck_h*LightSpeed)**3 * (4*np.pi * R_c**2 * g_nu)

    return flux_cooling*(1- j_kt(t))


def Flux_total(t, E_nu, *args):
    return Ue1_2 * Flux_total_e(t, E_nu, *args) + (1-Ue1_2) * Flux_total_x(t, E_nu, *args)


# Energy of anti-neutrino
def E_nu(E_e, c):
    result = (Delta_m+E_e)/(1-(E_e-p_e(E_e)*c)/m_p)
    return result


# Differential of E_nu to E_e 
def E_nu_at_E_e(E_e, c):
    
    result = (Delta_m+E_e)/(m_p*(1-(E_e-p_e(E_e)*c)/m_p)**2) + 1/(1-(E_e-p_e(E_e)*c)/m_p)
    return result



# Signal rate of Kamiokande per s-1 Mev-1 rad-1 
def SR_K(t, E_e, c, *args):

    R_c, T_c, tau_c, M_a, T_a, tau_a, m_phi, lambda_nu = args

    eff_K = 0.93 - np.exp(-(E_e/9)**2.5)       

    ret = N_K * cs_at_c(c, E_nu(E_e, c)) * Flux_total(t, E_nu(E_e, c), R_c, T_c, tau_c, M_a, T_a, tau_a, m_phi, lambda_nu) * E_nu_at_E_e(E_e, c) * eff_K

    return ret
    


# Signal rate of IMB per s-1 Mev-1 rad-1 
def SR_I(t, E_e, c, *args):

    R_c, T_c, tau_c, M_a, T_a, tau_a, m_phi, lambda_nu= args

    eff_I = np.clip(0.3975*(E_e/10)- 0.02625*(E_e/10)**2 - 0.59, None, 0.95)

    epsilon = 1 + 0.1*c     # Angular bias of IMB

    ret = N_I * cs_at_c(c, E_nu(E_e, c)) * Flux_total(t, E_nu(E_e, c), R_c, T_c, tau_c, M_a, T_a, tau_a, m_phi, lambda_nu) * E_nu_at_E_e(E_e, c) * eff_I * epsilon

    return ret


# Signal rate of Baksan per s-1 Mev-1 rad-1 
def SR_B(t, E_e, c, *args):

    R_c, T_c, tau_c, M_a, T_a, tau_a, m_phi, lambda_nu= args

    eff_B = 0.8

    ret = N_B * cs_at_c(c, E_nu(E_e, c)) * Flux_total(t, E_nu(E_e, c), R_c, T_c, tau_c, M_a, T_a, tau_a, m_phi, lambda_nu) * E_nu_at_E_e(E_e, c) * eff_B

    return ret

def SR_all(t, E, c, *theta):

    return (np.where(E<=7, 0, SR_K(t, E, c, *theta))
            +np.where(E<=19, 0, SR_I(t, E, c, *theta))
            +np.where(E<=12, 0, SR_B(t, E, c, *theta)))

if __name__ == '__main__':

    log_scale = True

    ranges_K: list = [(0, 30), (4, 50),  (-1, 1)]

    fix_0 = [10.70, 5.47, 4.63, 0.40, 2.08, 0.65, -4, -12]

    fix_1 = [10.89, 5.43, 4.67, 0.39, 2.08, 0.69, 1.63, -7.18]

    fix_2 = [10.89, 5.43, 4.67, 0.39, 2.08, 0.69, 1.63, -3]

    t_space = np.linspace(0, 10, 100)       # unit s
    E_space = np.linspace(7, 50, 300)

    # print(df.gl3_integrate(SR_K, ranges_K, args=fix_0))
    # print(df.gl3_integrate(SR_K, ranges_K, args=fix_1))
    # print(df.gl3_integrate(SR_K, ranges_K, args=fix_2))

    fig, ax = plt.subplots()

    # ranges_K: list = [(0, time_limit('max')), (4, 50),  (-1, 1)]
    # fun1_K = lambda x: SR_K(Data.Kam_t_3s, x, Data.Kam_c_3s, *fix_1) * Errors.Error_E(x, Data.Kam_E_3s, Data.Kam_dE_3s)
    # print(-df.gl3_integrate(SR_K, ranges_K, args=fix_1))
    # print(np.sum(np.log(df.gl_integrate(fun1_K, 4, 40) + Data.Kam_B_2009_3s/2)))

    # plt.plot(E_space, SR_all(1, E_space, 0, *fix_0), 'k-', label = 'orginal')
    # plt.plot(E_space, SR_all(1, E_space, 0,  *fix_1), 'r--', label = 'best fit')
    # plt.plot(E_space, SR_all(1, E_space, 0, *fix_2), 'b-.', label = 'most attenuated')

    y_e_1 = df.gl_integrate(lambda x: Flux_total_e(t_space, x, *fix_0), 7, 50)
    y_x_1 = df.gl_integrate(lambda x: Flux_total_x(t_space, x, *fix_0), 7, 50)
    y_total_1 = df.gl_integrate(lambda x: Flux_total(t_space, x, *fix_0), 7, 50)

    ax.plot(t_space, y_e_1, color = "#A682E8", linestyle ='-.', label = r'1D: $\bar\nu_e$ flux')
    ax.plot(t_space, y_x_1, color = "#A682E8",linestyle ='--', label = r'1D: $\bar\nu_x$ flux')
    ax.plot(t_space, y_total_1, color = "#A682E8", linestyle = '-', label = r'1D: Mixing flux')

    t_space_2 = np.linspace(0, time_limit('max'), 100)

    y_e_2 = df.gl_integrate(lambda x: flux_emulator_e(t_space_2, x), 7, 50)
    y_x_2 = df.gl_integrate(lambda x: flux_emulator_x(t_space_2, x), 7, 50)
    y_total_2 =  Ue1_2 * y_e_2 + (1-Ue1_2) * y_x_2

    ax.plot(t_space_2, y_e_2, color = "#006FED", linestyle ='-.', label = r'2D: $\bar\nu_e$ flux')
    ax.plot(t_space_2, y_x_2, color = "#006FED",linestyle ='--', label = r'2D: $\bar\nu_x$ flux')
    ax.plot(t_space_2, y_total_2, color = "#006FED", linestyle = '-', label = r'2D: Mixing flux')

    # fill x<tau_a area
    ax.fill_betweenx(np.array([0,2e10]), -0.1, 0.65, color = 'red', alpha = 0.1)

    ax.vlines(0.65, ymin=0, ymax=3e10, color='red', linestyle='--', label = r'$\tau_a$')
    ax.set_xlabel('Time (s)')

    ax.set_ylabel(r'Flux (cm$^{-2}$ s$^{-1}$)')
    # print(flux_2d(1, E_space, *fix_0))

    ax.set_ylim(0, 2e10)
    ax.set_xlim(0, time_limit('max'))
    plt.legend()

    plt.savefig(r'F:/Neutrino_SI/Plots/flux_osc.pdf')
    plt.show()

