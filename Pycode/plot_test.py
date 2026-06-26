import corner
import matplotlib.pyplot as plt
import numpy as np

effective_data_file = [
    '15.01', '15.05', '16.43', '16.65', '16.99', '17.00', '17.07', '17.10', '17.40', 
    '17.48', '17.50', '17.51', '17.83', '18.04', '18.05', '18.09', '18.10', '18.50',
    '19.02', '19.56', 
    # '19.83', 
    '19.99'
]

for file_name in effective_data_file:
    path_2d = f'F:/Neutrino_SI/Bin/smellycat260422/data/2d_m{file_name}_1e6_p32_log.npy'
    arr_2d = np.load(path_2d) # shape (32000, 2)
    fig_2d = corner.corner(arr_2d, labels=[r'log($m_{\Phi}$)', r'log(${\lambda}_{\Phi \nu}$)'], quantiles=[0.025, 0.5, 0.975], show_titles=True, title_fmt=".2f", color="black", smooth=1, plot_datapoints=False)
    plt.savefig(f'F:/Neutrino_SI/Bin/smellycat260422/figures/2d_m{file_name}_1e6_p32_log.png')
    plt.close(fig_2d)

path_1d = f'F:/Neutrino_SI/Bin/smellycat260422/data/1d_8p_5e5_p32_log_new.npy'
arr_1d = np.load(path_1d)[:,-2:] # shape (32000, 2)
fig_1d = corner.corner(arr_1d, labels=[r'log($m_{\Phi}$)', r'log(${\lambda}_{\Phi \nu}$)'], quantiles=[0.025, 0.5, 0.975], show_titles=True, title_fmt=".2f", color="black", smooth=1, plot_datapoints=False)
plt.savefig(f'F:/Neutrino_SI/Bin/smellycat260422/figures/1d_8p_5e5_p32_log_new.png')
plt.close(fig_1d)