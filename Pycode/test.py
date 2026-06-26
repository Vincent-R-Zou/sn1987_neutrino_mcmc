import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples, plots

# 1. Define your files and labels
effective_data_file = [
    '15.01', '15.05', '16.43', '16.65', '16.99', '17.00', '17.07', '17.10', '17.40', 
    '17.48', '17.50', '17.51', '17.83', '18.04', '18.05', '18.09', '18.10', '18.50',
    '19.02', '19.56', '19.83', '19.99'
]

param_names = ['logm', 'loglam']
param_labels = [r'\log(m_\phi/\mathrm{eV})', r'\log(\lambda_{\phi\nu})']


X_LIMITS = [-1.0, 4.0]  
Y_LIMITS = [-10.0, -3.0] 


mc_samples_list = []
titles = []
stacked_list = []

for file_id in effective_data_file:
    path = f'F:/Neutrino_SI/Bin/smellycat260608/data/2d_m{file_id}_1e6_p32_log.npy'
    arr = np.load(path)
    stacked_list.append(arr)
    sample = MCSamples(samples=arr, names=param_names, labels=param_labels, 
                       settings={'smooth_scale_2D': -1})
    mc_samples_list.append(sample)
    titles.append(rf'$M = {file_id} M_\odot$')

stacked_arr = np.concatenate(stacked_list, axis=0)
stacked_sample = MCSamples(samples=stacked_arr, names=param_names, labels=param_labels,)

mc_samples_list.append(stacked_sample)
titles.append('Stacked') # Title for the 23rd plot

fig, axes = plt.subplots(nrows=4, ncols=6, 
                        #  figsize=(20, 13), 
                         constrained_layout=True)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

axes_flat = axes.flatten()
g = plots.get_single_plotter()
# g.settings.axes_labelsize = 22
# g.settings.axes_fontsize = 20
# g.settings.legend_fontsize = 25


for i, (title, sample) in enumerate(zip(titles, mc_samples_list)):
    ax = axes_flat[i]
    
    plot_color = '#c44e52' if i == 22 else "#006FED" 
    g.plot_2d(sample, 'logm', 'loglam', ax=ax, filled=True, colors=[plot_color])
    
    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top')

    # ax.set_title(title, fontsize=14, fontweight='bold' if i == 22 else 'normal')
    
    # ENFORCE GLOBAL LIMITS HERE
    # if i <= 21:
    #     if effective_data_file[i] in ['15.01', '15.05', '17.40', '17.48', '18.09']:
    ax.set_xlim(X_LIMITS)
    ax.set_ylim(Y_LIMITS)
    

    if i % 6 != 0:
        ax.set_ylabel('')
    if i < 18:
        ax.set_xlabel('')


ax_legend = axes_flat[23]
ax_legend.axis('off') # Turn off the box and ticks


from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#006FED", alpha=0.7, label='Individual Mass Spectra'),
    Patch(facecolor='#c44e52', alpha=0.7, label='Stacked Constraint')
]
ax_legend.legend(handles=legend_elements, loc='center', frameon=False, fontsize=18)

plt.savefig('24_panel_grid_with_limits.pdf')

plt.show()