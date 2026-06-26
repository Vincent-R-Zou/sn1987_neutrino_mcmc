import SignalRate_2d_osc
import SignalRate_2d
import numpy as np
from scipy.optimize import minimize
import definition as df
import matplotlib.pyplot as plt


params = [0, 0, 0, 0, 0, 0, 4, -12]

def minimize_sr(vars):
    t, E = vars
    return - SignalRate_2d.SR_K(t,E,0,*params)

ret = minimize(minimize_sr, x0=[0.1, 10], bounds=((0.01, 4), (4, 50)), method='L-BFGS-B')
print(-ret.fun, ret.x)

f_max = -ret.fun

t_range = (0.01, 4)
E_range = (4, 50)
num_samples = 100

def sample_rejection(flux_func, E_range, t_range, f_max, num_samples, *args):
    samples = []
    
    while len(samples) < num_samples:
        # 1. Propose E and t uniformly within your bounds
        E_prop = np.random.uniform(E_range[0], E_range[1])
        t_prop = np.random.uniform(t_range[0], t_range[1])
        
        # 2. Draw a uniform height to test against the flux
        u = np.random.uniform(0, f_max)
        
        # 3. Accept if the test height is under the distribution surface
        if u < flux_func(t_prop, E_prop, 0, *args):
            samples.append((t_prop, E_prop))
            
    return np.array(samples)

t_samples, E_samples = sample_rejection(SignalRate_2d.SR_K, E_range, t_range, f_max, num_samples, *params).T

print(t_samples)

# plot 2d histogram of samples
plt.hist2d(t_samples, E_samples, bins=20, range=[t_range, E_range], cmap='Blues')
plt.colorbar(label='Count in bin')
plt.xlabel('Time (s)')
plt.ylabel('Energy (MeV)')
plt.title('Rejection Sampling of SR_K')
plt.show()