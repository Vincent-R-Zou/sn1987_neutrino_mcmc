import definition as df
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import neutrino_new
import neutrino_new_log



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
def Flux_total(t, E_nu, *args):

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

    # t_cool = np.clip(t-tau_a, 0, None)      # make sure return 0 when t<tau_a

    t_cool = t-tau_a

    T_ct = T_c*np.exp(-t_cool/4/tau_c)

    g_nu = E_nu**2/(1+df.safe_exp(E_nu/T_ct))
    
    # flux_cooling = 1/(4*np.pi*Distance_SN1987**2) * np.pi * LightSpeed/ (Planck_h*LightSpeed)**3 * (4*np.pi * R_c**2 * g_nu) * np.sign(t_cool)      # make sure return 0 when t<tau_a

    flux_cooling = 1/(4*np.pi*Distance_SN1987**2) * np.pi * LightSpeed/ (Planck_h*LightSpeed)**3 * (4*np.pi * R_c**2 * g_nu)

    flux = flux_accretion + (1 - j_kt(t)) *flux_cooling

    if log_scale:
        return flux * np.exp(neutrino_new_log.d_phi(E_nu, m_phi, lambda_nu))
    else:
        return flux * np.exp(neutrino_new.d_phi(E_nu, m_phi, lambda_nu))



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

    # eff_K = 0.932/np.sqrt(1+(34/(12-7*E_e+E_e**2))**2)        # This one is not well-fitted

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

    # eff_B = 0.8*(1-np.exp(-(E_e/10.17)**9))

    eff_B = 0.8

    ret = N_B * cs_at_c(c, E_nu(E_e, c)) * Flux_total(t, E_nu(E_e, c), R_c, T_c, tau_c, M_a, T_a, tau_a, m_phi, lambda_nu) * E_nu_at_E_e(E_e, c) * eff_B

    return ret


if __name__ == '__main__':

    log_scale = False


    fix_0 = [15, 5, 5, 0.5, 2, 0.3, 1e-6, 1e-6]

    fix_1 = [15, 5, 5, 0.5, 2, 0.3, 200, 200]

    fix_2 = [30, 4.9, 4.7, 1, 2.33, 0.54, 100, 100]
    
    E_range = (5, 50)
    t_range = (0.01, 30)
    range_k = [t_range, E_range,  (-1, 1)]