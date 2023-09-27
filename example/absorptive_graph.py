from json import load
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.colors import LightSource

MODE_NUM = 100

# avg_powers = np.array([3])
# avg_powers = 10**( avg_powers /10)/1000
avg_powers = [0.001059253725177289]
file = open("results/new_graphing/visibilities.txt", "a+")

for power in avg_powers:
    print("file name is:", f"absorptive{power}.json")
    filename = f"results/archived/all_Raman_long_450_power_many_photons/absorptive{power}.json"
    data = load(open(filename))

    direct_results = data["direct results"]
    bs_results = data["bs results"]

    # acquire direct detection results
    bins = np.zeros(4)
    for trial in direct_results:
        bins += np.array(trial["counts"])
    bins *= (1 / sum(bins))

    # acquire bs results
    num = data["num_phase"]
    phases = np.linspace(0, 2*np.pi, num=num)
    freq_0 = []
    freq_1 = []
    st_dev_0 = []
    st_dev_1 = []
    # Iterate over one phase angle, which is a list num_trials (per phase) dictionaries 
    for res in bs_results:
        counts = []
        total = 0
        # Looking at each such dictionary: (every trial)

        # TRYING
        # TRYING_total = len(res) * MODE_NUM
        
        for trial in res:
            # We are looking at the cases where the entanglement was heralded by a detection at detector1
            counts.append(trial["counts1"])
            total += trial["total_count1"]
        counts = np.array(counts, dtype=float)
    
        # Here, the axis simply states that all 1st elements of sublists should be sum/std
        # together and all 2nd elements should be sum/std together. 
        st_dev = np.std(counts, axis=0) 
        # average
        counts = np.sum(counts, axis=0)

        # THESE MJUST BE total
        counts *= (1 / total)
        st_dev *= (1 / (total/len(res)))
        # THESE MJUST BE total

        print("std:", st_dev, "mean:", counts)
        
        freq_0.append(counts[0])
        freq_1.append(counts[1])
        st_dev_0.append(st_dev[0])  # mult. by 2, so have 1 on top and bottom of point
        st_dev_1.append(st_dev[1])

    # curve fitting
    def sin(x, A, phi, z): return A*np.sin(x + phi) + z
    params_0, _ = curve_fit(sin, phases, freq_0)
    params_1, _ = curve_fit(sin, phases, freq_1)
    vis_0 = abs(params_0[0]) / abs(params_0[2])
    vis_1 = abs(params_1[0]) / abs(params_1[2])

    # off-diagonal density matrix
    vis_total = (vis_0 + vis_1) / 2
    off_diag = vis_total * (bins[1] + bins[2]) / 2

    # plotting density matrix
    # plot the 2-d matrix
    plt.rc('font', size=10)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=15)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('legend', fontsize=15)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    ax.view_init(azim=-30, elev=20)

    offset = 3
    total = list(bins) + [off_diag, off_diag]
    height = np.log10(total) + offset
    width = depth = 0.8
    x = np.array([0, 1, 2, 3, 1, 2]) - (width/2)
    y = np.array([0, 1, 2, 3, 2, 1]) - (width/2)
    color = (['tab:blue'] * 4) + (['tab:orange'] * 2)
    ls = LightSource(100, 60)
    ax.bar3d(x, y, np.zeros(6), width, depth, height,
            color=color, edgecolor='black', shade=True, lightsource=ls)

    def log_tick_formatter(val, pos=None): return f"$10^{{{int(val-offset)}}}$"
    ax.zaxis.set_major_formatter(tck.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_locator(tck.MaxNLocator(integer=True))

    # ax.set_title(r'Reconstructed $|\tilde{\rho}|$')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticklabels([r'$|00\rangle$', r'$|01\rangle$', r'$|10\rangle$', r'$|11\rangle$'])
    ax.set_yticklabels([r'$\langle00|$', r'$\langle01|$', r'$\langle10|$', r'$\langle11|$'])
    plt.savefig('density.png', bbox_inches='tight')
    plt.show()

    # plotting interference
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot()

    ax.errorbar(phases/np.pi, freq_0, yerr=st_dev_0, marker='o', ls='', capsize=5, label=r'$p_{01}$')
    ax.errorbar(phases/np.pi, freq_1, yerr=st_dev_1, marker='o', ls='', capsize=5, label=r'$p_{10}$')
    ax.plot(phases/np.pi, np.zeros(phases.shape), ls='--', color='gray')
    plt.plot(phases/np.pi, sin(phases, *params_0), color='tab:blue', ls='--')
    plt.plot(phases/np.pi, sin(phases, *params_1), color='tab:orange', ls='--')

    ax.set_title("Photon Interference")
    ax.xaxis.set_major_formatter(tck.FormatStrFormatter(r'%g $\pi$'))
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=0.5))
    ax.set_xlabel("Relative Phase")
    ax.set_ylabel("Detection Rate")
    ax.legend()
    fig.tight_layout()
    plt.savefig(f'results/new_graphing/interference_{power}.png')
    plt.show()

    # output
    print("Detector 1 visibility:", vis_1)
    print("Detector 2 visibility:", vis_0)
    print("Combined:", vis_total)
    print("Off-diagonal:", off_diag)
    file.write(f"\n{vis_total}")
file.close()



# from json import load
# import numpy as np
# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt
# import matplotlib.ticker as tck
# from matplotlib.colors import LightSource

# MODE_NUM = 100

# # avg_powers = np.array([3])
# # avg_powers = 10**( avg_powers /10)/1000
# avg_powers = [0.001059253725177289]
# file = open("results/new_graphing/rate_averaged_visibilities.txt", "a+")

# for power in avg_powers:
#     print("file name is:", f"absorptive{power}.json")
#     filename = f"results/archived/all_Raman_long_450_power_many_photons/absorptive{power}.json"
#     data = load(open(filename))

#     direct_results = data["direct results"]
#     bs_results = data["bs results"]

#     # acquire direct detection results
#     bins = np.zeros(4)
#     for trial in direct_results:
#         bins += np.array(trial["counts"])
#     bins *= (1 / sum(bins))

#     # acquire bs results
#     num = data["num_phase"]
#     phases = np.linspace(0, 2*np.pi, num=num)
#     freq_0 = []
#     freq_1 = []
#     st_dev_0 = []
#     st_dev_1 = []
#     # Iterate over one phase angle, which is a list num_trials (per phase) dictionaries
#     # 
#     # How it used to work:
#     # You take a phase angle, look at all the trials' total_counts (no. of heralded entanglements) and the counts (heraded entangements where we also saw interference detection,
#     # basically, interference measurement counts conditioned on succesful entanglement due to heralding)
#     # You find the std and mean of the counts by finding the np.std and np.sum of the counts values themselves and divinding them by the sum of total_counts for all the trials in the 
#     # phase angle.

#     # The new method: We will find the rate of succesful interference measurent for every trial and then find the rates' std and means directly. This could give is a better result. 
#     for res in bs_results:
#         # counts = []
#         # total = 0
#         rates = []
#         # Looking at each such dictionary: (every trial)

#         # TRYING
#         # TRYING_total = len(res) * MODE_NUM
        
#         for trial in res:
#             # We are looking at the cases where the entanglement was heralded by a detection at detector1
#             # counts.append(trial["counts1"])
#             # total += trial["total_count1"]
#             rates.append(np.array(trial["counts1"]) / trial["total_count1"])
#         # rates = np.array(rates, dtype=float)
#         print("len of rates:", len(rates))
#         rate_std_dev = np.std(rates, axis  = 0)
#         rate_mean = np.mean(rates, axis = 0)
#         assert len(rate_mean) == 2
#         assert len(rate_std_dev) == 2
#         # Here, the axis simply states that all 1st elements of sublists should be sum/std
#         # together and all 2nd elements should be sum/std together. 
#         # st_dev = np.std(counts, axis=0) 
#         # average
#         # counts = np.sum(counts, axis=0)

#         # THESE MUST BE total
#         # counts *= (1 / total) # Here is where we calculate the average rate. total inter. meas / total bsm. Instead, we will be doing avg(inter. meas / bsm)
#         # st_dev *= (1 / np.sqrt(total))
#         # THESE MJUST BE total
 
#         print("std:", rate_std_dev, "mean:", rate_mean)
        
#         freq_0.append(rate_mean[0])
#         freq_1.append(rate_mean[1])
#         st_dev_0.append(rate_std_dev[0])  # mult. by 2, so have 1 standard devn above and below mean in error bar. 
#         st_dev_1.append(rate_std_dev[1])

#     # curve fitting
#     def sin(x, A, phi, z): return A*np.sin(x + phi) + z
#     params_0, _ = curve_fit(sin, phases, freq_0)
#     params_1, _ = curve_fit(sin, phases, freq_1)
#     vis_0 = abs(params_0[0]) / abs(params_0[2])
#     vis_1 = abs(params_1[0]) / abs(params_1[2])

#     # off-diagonal density matrix
#     vis_total = (vis_0 + vis_1) / 2
#     off_diag = vis_total * (bins[1] + bins[2]) / 2

#     # plotting density matrix
#     # plot the 2-d matrix
#     plt.rc('font', size=10)
#     plt.rc('axes', titlesize=18)
#     plt.rc('axes', labelsize=15)
#     plt.rc('xtick', labelsize=15)
#     plt.rc('ytick', labelsize=15)
#     plt.rc('legend', fontsize=15)

#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(projection="3d")
#     ax.view_init(azim=-30, elev=20)

#     offset = 3
#     total = list(bins) + [off_diag, off_diag]
#     height = np.log10(total) + offset
#     width = depth = 0.8
#     x = np.array([0, 1, 2, 3, 1, 2]) - (width/2)
#     y = np.array([0, 1, 2, 3, 2, 1]) - (width/2)
#     color = (['tab:blue'] * 4) + (['tab:orange'] * 2)
#     ls = LightSource(100, 60)
#     ax.bar3d(x, y, np.zeros(6), width, depth, height,
#             color=color, edgecolor='black', shade=True, lightsource=ls)

#     def log_tick_formatter(val, pos=None): return f"$10^{{{int(val-offset)}}}$"
#     ax.zaxis.set_major_formatter(tck.FuncFormatter(log_tick_formatter))
#     ax.zaxis.set_major_locator(tck.MaxNLocator(integer=True))

#     # ax.set_title(r'Reconstructed $|\tilde{\rho}|$')
#     ax.set_xticks([0, 1, 2, 3])
#     ax.set_yticks([0, 1, 2, 3])
#     ax.set_xticklabels([r'$|00\rangle$', r'$|01\rangle$', r'$|10\rangle$', r'$|11\rangle$'])
#     ax.set_yticklabels([r'$\langle00|$', r'$\langle01|$', r'$\langle10|$', r'$\langle11|$'])
#     plt.savefig('density.png', bbox_inches='tight')
#     plt.show()

#     # plotting interference
#     fig = plt.figure(figsize=(8, 5))
#     ax = fig.add_subplot()

#     ax.errorbar(phases/np.pi, freq_0, yerr=st_dev_0, marker='o', ls='', capsize=5, label=r'$p_{01}$')
#     ax.errorbar(phases/np.pi, freq_1, yerr=st_dev_1, marker='o', ls='', capsize=5, label=r'$p_{10}$')
#     ax.plot(phases/np.pi, np.zeros(phases.shape), ls='--', color='gray')
#     plt.plot(phases/np.pi, sin(phases, *params_0), color='tab:blue', ls='--')
#     plt.plot(phases/np.pi, sin(phases, *params_1), color='tab:orange', ls='--')

#     ax.set_title("Photon Interference")
#     ax.xaxis.set_major_formatter(tck.FormatStrFormatter(r'%g $\pi$'))
#     ax.xaxis.set_major_locator(tck.MultipleLocator(base=0.5))
#     ax.set_xlabel("Relative Phase")
#     ax.set_ylabel("Detection Rate")
#     ax.legend()
#     fig.tight_layout()
#     plt.savefig(f'results/new_graphing/rate_averaged_interference_{power}.png')
#     plt.show()

#     # output
#     print("Detector 1 visibility:", vis_1)
#     print("Detector 2 visibility:", vis_0)
#     print("Combined:", vis_total)
#     print("Off-diagonal:", off_diag)
#     file.write(f"\n{vis_total}")
# file.close()
