import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import make_interp_spline, RectBivariateSpline
import matplotlib.gridspec as gridspec
from scipy.special import gamma
from scipy.integrate import quad
from scipy.integrate import quad_vec
import definition

# from setup import index

effective_data_file = ['15.01', '15.05', '16.43', '16.65', '16.99', '17.00', '17.07', '17.10', '17.40', '17.48', '17.50', '17.51', '17.83', '18.04', '18.05', '18.09', '18.10', '18.50', '19.02', '19.56', '19.83', '19.99']  
# in unit of solar mass, corresponding to the 22 data files

file_str = effective_data_file[0] 
# index from 0 to 21, corresponding to the 22 data files

D = definition.Distance_SN1987 # Distance to SN 1987A in cm
filename_1= "F:/Neutrino_SI/Bin/2D_trim_data/"+ file_str +"/nuspec.1.xg"   # r"../2D_trim_data/" on sc
filename_2 = "F:/Neutrino_SI/Bin/2D_trim_data/"+ file_str +"/nuspec.2.xg"

def read_xg(filename):

    data = []
    current_time = None

    with open(filename) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # time line
            if line.startswith("#"):
                current_time = float(line.split()[-1])
                continue

            # data line
            E, EdFdE = map(float, line.split())

            data.append({
                "time": current_time,
                "E": E,
                "EdFdE": EdFdE
            })

    df = pd.DataFrame(data)

    return df


df_e = read_xg(filename_1)
E_space_2d = np.linspace(4, 50, 1000)
times_space_e = np.array(df_e["time"].unique()) # 4139 times
energies_space_e = np.array(df_e["E"]).reshape(len(times_space_e), 12) # 49668    energies grouped into 4139 times, 12 per each time

df_x = read_xg(filename_2)
times_space_x = np.array(df_x["time"].unique())
energies_space_x = np.array(df_x["E"]).reshape(len(times_space_x), 12)


Eflux_e = np.array(df_e["EdFdE"]).reshape(len(times_space_e), 12)
flux_e = Eflux_e * 1e75 * 624150.9074461 / energies_space_e / (4 * np.pi * D**2)

Elux_x = np.array(df_x["EdFdE"]).reshape(len(times_space_x), 12)
flux_x = Elux_x * 1e75 * 624150.9074461 / energies_space_x / (4 * np.pi * D**2) / 4 
# Assuming x-flavor neutrinos are equally distributed among 4 flavors (nu_mu, nu_tau, anti-nu_mu, anti-nu_tau)

grid_F_e = np.array([make_interp_spline(energies_space_e[i, :], flux_e[i, :], k=3)(E_space_2d) for i in range(len(times_space_e))])

grid_F_x = np.array([make_interp_spline(energies_space_x[i, :], flux_x[i, :], k=3)(E_space_2d) for i in range(len(times_space_x))])

flux_interpolator_e = RectBivariateSpline(times_space_e, E_space_2d, grid_F_e, kx=3, ky=3)
flux_interpolator_x = RectBivariateSpline(times_space_x, E_space_2d, grid_F_x, kx=3, ky=3)

# 5. Define your incredibly fast emulator function

def flux_emulator_e(t, E, grid=False):
    """
    Evaluates the pre-computed 2D spline for anti-nu_e flavor.
    """
    return np.clip(flux_interpolator_e(t, E, grid=grid), 0, None) # Ensure non-negative flux


def flux_emulator_x(t, E, grid=False):
    """
    Evaluates the pre-computed 2D spline for x-flavor.
    """
    return np.clip(flux_interpolator_x(t, E, grid=grid), 0, None) 


def time_limit(type):
    if type == "min":
        return np.max([times_space_e[0], times_space_x[0]])
    elif type == "max":
        return np.min([times_space_e[-1], times_space_x[-1]])



if __name__ == "__main__":

    t_vals = np.linspace(times_space_e.min(), times_space_x.max(), 2000)
    E_vals = np.linspace(4, 60, 1000)

    print(time_limit("min"), time_limit("max"))
    # 2d projection

    plt.plot(t_vals, flux_emulator_e(t_vals, 30), label = str(file_str) + r"$M_{\odot}$", color='blue')
    plt.plot(t_vals, flux_emulator_x(t_vals, 30), label = str(file_str) + r"$M_{\odot}$", color='red')
    plt.xlabel("Time after bounce(s)")
    plt.ylabel(r"Differential flux ($MeV^{-1}s^{-1}cm^{-2}$)")
    plt.title(r"$E_\nu$=30 MeV" ) 
    plt.show()


    # 3d projection

    # TT, EE = np.meshgrid(t_vals, E_vals, indexing="ij")
    # ZZ = flux_emulator(t_vals, E_vals, grid=True)

    # fig = plt.figure()

    # ax = fig.add_subplot(111, projection="3d")

    # surf = ax.plot_surface(EE, TT, ZZ,
    #                     cmap=cm.coolwarm,
    #                     edgecolor='none',
    #                     linewidth=0,
    #                     antialiased=False,
    #                     # rstride=20,cstride=10,
    #                     ) # define the resolution

    # cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20)
    # cbar.set_label("Differential Flux ($MeV^{-1}s^{-1}cm^{-2}$)")

    # ax.set_title("2-D Cubic Interpolation of Differential Flux in Time and Energy")
    # ax.set_xlabel("Energy (MeV)")
    # ax.set_ylabel("Time after bounce (s)")
    # # ax.set_zlabel("Differential Flux ($MeV^{-1}s^{-1}$)")
    # ax.tick_params(axis='x')
    # ax.tick_params(axis='y')
    # ax.tick_params(axis='z')
    # # ax.zaxis.get_offset_text().set_fontsize(12)
    # # cbar.ax.tick_params(labelsize=12)
    # # cbar.ax.yaxis.get_offset_text().set_fontsize(12)
    # # ax.set_xlim(0, 40)
    # # ax.set_ylim(-0.35, 4)
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # white
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # # ax.xaxis._axinfo["grid"]["linewidth"] = 0.5
    # # ax.yaxis._axinfo["grid"]["linewidth"] = 0.5
    # # ax.zaxis._axinfo["grid"]["linewidth"] = 0.5
    # # ax.xaxis._axinfo["grid"]["linestyle"] = "--"
    # # ax.yaxis._axinfo["grid"]["linestyle"] = "--"
    # # ax.zaxis._axinfo["grid"]["linestyle"] = "--"
    # ax.view_init(elev=30, azim=-60)
    # plt.show()

