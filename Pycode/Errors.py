import numpy as np

def Error_E(E,Ei,Delta_E):
    ret = 1/(np.sqrt(2*np.pi)*Delta_E) * np.exp(-(E-Ei)**2/(2*Delta_E**2))
    return ret