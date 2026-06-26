import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import matplotlib.pyplot as plt
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.integrate import nquad, quad
import xlwt
import warnings

m_nu = 1e-8       # neutrino mass in eV
Ue1_2 = 0.7         # |U_e1|^2, the mixing matrix element for electron neutrino
T_nu = 2.725 * 0.7138 * 8.617e-5    # 1.6761 e-4 eV

Distance_SN1987 = 50 * 1e3 * 3.0857e16 * 1e2        # Distance of SN 1987A in unit cm

E_max_1d = 50
t_max_1d = 30
E_max_2d = 50

E_min_K = 7     # energy threshold of Kamiokande-II in unit MeV
E_min_I = 19    # energy threshold of IMB
E_min_B = 12    # energy threshold of Baksan

N_K = 1.43e32   # number of protons in Kamiokande-II
N_I = 4.55e32   # number of protons in IMB
N_B = 1.87e31   # number of protons in Baksan


m_e = 0.511     # electron mass in unit MeV
m_p = 938.272       # proton mass
m_n = 939.565       # neutron mass
Planck_h = 6.62e-34 / 1.60e-19 / 1e6        # Planck constant in unit Mev·s 
LightSpeed = 2.99792458e8*1e2       # light speed unit cm/s
delta_m = (m_n**2-m_p**2-m_e**2)/(2*m_p)        
Delta_m = m_n - m_p     # namely 1.293 MeV
f_d_IMB = 0.9055    # live time fraction of IMB
tau_d_IMB = 0.0350      # dead time fraction of IMB
m_sun = 1.989e30


def safe_exp(x):
    """safe exponential function to avoid overflow warning"""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'overflow')
        return np.exp(x)
    

def integrate(Fun, lower_limit, upper_limit, *args, **kwargs):

    precision = 300    
    x = np.linspace(lower_limit, upper_limit, precision)
    y = Fun(x, *args,**kwargs)
    dx = (upper_limit - lower_limit) / precision
    result = (y * dx).sum(axis = 0)
    return result


def integrate_vec(Fun, lower_limit, upper_limit, args=()):
    precision_vec = 300     
    x = np.linspace(lower_limit, upper_limit, precision_vec)
    arr = Fun(x[0])
    X, Y= np.meshgrid(x, arr)
    y = Fun(X.T)
    dx = (upper_limit - lower_limit) / precision_vec
    result = (y * dx).sum(axis = 0)
    return result


def stay_increase(arr):
    index, new_arr = [0], [arr[0]]
    for i in range(len(arr)-1):
        if arr[i] < arr[i+1]:
            index.append(i+1)
            new_arr.append(arr[i+1])
        else:
            arr[i+1] = arr[i]

    return np.array(new_arr), index


def save_by_index(arr, index):
    new_arr = []
    for index_i in index:
        new_arr.append(arr[index_i])
    return new_arr


def func_build(x, arr_a, arr_b, mode = 'curve'):
    arr_a_increase = stay_increase(arr_a)[0]
    index = stay_increase(arr_a)[1]
    arr_b_increase = save_by_index(arr_b, index)
    if mode == 'curve':
        spl = IUS(arr_a_increase, arr_b_increase)
        return spl(x)
    if mode == 'sline':
        return np.interp(x, arr_a_increase, arr_b_increase)


def readdata(i, path):
    df = pd.read_excel(path, usecols=[i], header=None)
    df_li = np.array(df.values.tolist()).flatten()
    return df_li

def gl2_integrate_vector(f_vec, bounds, n=64):
    """
    Double Gauss-Legendre integration for vectorized bounds.
    
    Returns an array matching the shape of the input bounds (e.g., a vector of N results).
    
    Parameters
    ----------
    f_vec : callable
        f(x, y). Evaluated on 3D broadcast grids.
    bounds : tuple
        ((ax, bx), (ay, by))
        Each element should broadcast to a common shape N_shape.
    n : int or tuple
        Number of quadrature points (nx, ny).
    """
    (ax, bx), (ay, by) = bounds
    ax, bx = np.asarray(ax), np.asarray(bx)
    ay, by = np.asarray(ay), np.asarray(by)
    
    N_shape = np.broadcast(ax, bx, ay, by).shape
    N = np.prod(N_shape) if N_shape else 1

    if isinstance(n, (tuple, list)):
        nx, ny = int(n[0]), int(n[1])
    else:
        nx = ny = int(n)

    gx, wx = leggauss(nx)
    gy, wy = leggauss(ny)


    # Shape: (nx, 1, 1)
    gx_b = gx.reshape(nx, 1, 1)
    wx_b = wx.reshape(nx, 1, 1)
    
    # Shape: (1, ny, 1)
    gy_b = gy.reshape(1, ny, 1)
    wy_b = wy.reshape(1, ny, 1)

    expand_bounds = lambda arr: np.broadcast_to(arr, N_shape).reshape(1, 1, N)
    
    ax_b, bx_b = expand_bounds(ax), expand_bounds(bx)
    ay_b, by_b = expand_bounds(ay), expand_bounds(by)

    half_width_x = 0.5 * (bx_b - ax_b)
    half_width_y = 0.5 * (by_b - ay_b)
    
    center_x = 0.5 * (bx_b + ax_b)
    center_y = 0.5 * (by_b + ay_b)

    X = half_width_x * gx_b + center_x
    Y = half_width_y * gy_b + center_y

    F = f_vec(X, Y)

    WX = wx_b * half_width_x 
    WY = wy_b * half_width_y
    
    weighted_F = F * WX * WY
    
    val = np.sum(weighted_F, axis=(0, 1)) 

    val = val.reshape(N_shape)

    if val.ndim == 0:
        return val.item()
        
    return val


def gl_integrate(Fun, lower_limit, upper_limit, precision=64, *args, **kwargs):
    """
    gauss-legendre integration for 1D function
    """

    gx, wx = leggauss(precision) 

    a = np.asarray(lower_limit)
    b = np.asarray(upper_limit)
    x = 0.5 * (b - a) * gx[:, None] + 0.5 * (b + a)
    w = 0.5 * (b - a) * wx[:, None]                  

    y = Fun(x, *args, **kwargs) 

    result = (w * y).sum(axis=0)
    return result


def gl3_integrate(f_vec, 
                  bounds, 
                  n=64, 
                  args=()):
    """
    3D Gauss-Legendre integration for vectorized bounds.
    """
    
    (ax,bx), (ay,by), (az,bz) = bounds
  
    gx, wx = leggauss(n)
    gy, wy = leggauss(n)
    gz, wz = leggauss(n)

    x = 0.5*(bx-ax)*gx + 0.5*(bx+ax)
    y = 0.5*(by-ay)*gy + 0.5*(by+ay)
    z = 0.5*(bz-az)*gz + 0.5*(bz+az)
    wx = wx * 0.5*(bx-ax)
    wy = wy * 0.5*(by-ay)
    wz = wz * 0.5*(bz-az)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    F = f_vec(X, Y, Z, *args)  # 形状 (n, n, n)

    tmp = F * wz[np.newaxis, np.newaxis, :]
    tmp = tmp.sum(axis=2)
    tmp = tmp * wy[np.newaxis, :]
    tmp = tmp.sum(axis=1)
    tmp = tmp * wx[:]
    val = tmp.sum()
    return val


def gl2_integrate(f_vec, 
                  bounds, 
                  n=64, 
                  args=()):
    """
    2D Gauss-Legendre integration for vectorized bounds.
    """
    (ax, bx), (ay, by) = bounds

    gx, wx = leggauss(n)
    gy, wy = leggauss(n)

    x = 0.5*(bx-ax)*gx + 0.5*(bx+ax)
    y = 0.5*(by-ay)*gy + 0.5*(by+ay)
    wx = wx * 0.5*(bx-ax)
    wy = wy * 0.5*(by-ay)
  
    X, Y = np.meshgrid(x, y, indexing='ij')

    F = f_vec(X, Y, *args) 

    tmp = F * wy[np.newaxis, :]
    tmp = tmp.sum(axis=1)
    val = np.dot(tmp, wx)
    return val



def gl2_integrate_vec(f_vec, bounds, z, n=64, args=()):
    
    # translate the above description into English
    """
    two-dimensional Gauss–Legendre vectorized integration (x, y integration; z is an array)
    ---------------------------------------------------
    f_vec(x, y, z, *args) -> ndarray
        Must support broadcasting with x,y as arrays and z as an array of any shape,
        returning an array compatible with the broadcast result of (x,y,z).
        For example: x: (nx,1,...), y: (1,ny,...), z: (...), then f_vec returns (nx,ny,...).
    bounds = ((ax, bx), (ay, by))    # constant limits for x and y
    z : ndarray                      # any shape, result will match z's shape
    n : int or (nx,ny)               # number of Gauss–Legendre points per dimension
    args : tuple                     # other parameters, passed as-is to f_vec
    Returns: ∬ f(x,y,z, *args) dx dy, shape matches z
    """

    (ax, bx), (ay, by) = bounds
    if isinstance(n, (tuple, list)):
        nx, ny = int(n[0]), int(n[1])
    else:
        nx = ny = int(n)

    gx, wx = leggauss(nx)
    gy, wy = leggauss(ny)

    x = 0.5*(bx-ax)*gx + 0.5*(bx+ax)
    y = 0.5*(by-ay)*gy + 0.5*(by+ay)
    wx = wx * 0.5*(bx-ax)
    wy = wy * 0.5*(by-ay)

    X, Y = np.meshgrid(x, y, indexing='ij')  

    Z = np.asarray(z)
    expand = (None,)*Z.ndim
    Xb = X[(slice(None), slice(None)) + expand]  
    Yb = Y[(slice(None), slice(None)) + expand]  

    F = f_vec(Xb, Yb, Z, *args)

    tmp = np.tensordot(F, wy, axes=(1, 0))      # -> (nx, *Z.shape)
    val = np.tensordot(tmp, wx, axes=(0, 0))    # -> (*Z.shape)
    return val


def g_integrate(Fun, lower_limit, upper_limit, percision=64, *args, **kwargs):  
    gx, wx = leggauss(percision) 

    orig_shape = np.shape(upper_limit)           
    L, U = np.meshgrid(lower_limit, upper_limit) 

    x = 0.5*(U- L)*gx + 0.5*(U + L) 
    wx = wx * 0.5*(U - L) 

    ret = (wx*Fun(x, *args, **kwargs)).sum(axis = 1)  
    ret = ret.reshape(orig_shape)                     
    return ret


def save_mcmc_result(flat_samples, path):
    my_workbook = xlwt.Workbook()
    sheet = my_workbook.add_sheet('mcmc_result')
    for i in range(flat_samples.shape[0]):
        for j in range(flat_samples.shape[1]):
            sheet.write(i, j, flat_samples[i][j])
    my_workbook.save(path)



if __name__  == "__main__":


    def test_fun(x, y, z, model = 'gl'):

        ret = x**2 + test_y(y, model)**2 + z**2

        return ret
    

    def test_y(y, model):

        int_part = lambda x: np.sqrt(x)/(1+np.exp(x))

        if model == 'gl':

            ret = g_integrate(int_part, 0, y)
        
        elif model == 'fb':

            y_space = np.linspace(0, 10, 100)

            ret = func_build(y, y_space, g_integrate(int_part, 0, y_space))

        elif model == 'ni':
            
            ret = integrate(int_part, 0, y)
        
        elif model == 'sc':

            ret = quad(int_part, 0, y)[0]

        return ret
    
    

    xarray = np.array([4,5])

    print(nquad(test_fun, [(0,1), (0,1), (0,1)], args = ['gl'])[0])

    print(gl3_integrate(test_fun, [(0,1), (0,1), (0,1)], args = ['gl']))

    print(gl3_integrate(test_fun, [(0,1), (0,1), (0,1)], args = ["ni"]))







