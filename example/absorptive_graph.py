from json import load
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.colors import LightSource
from csv import writer
from copy import copy

def analyze_data(data, sin):
    direct_results = data["direct results"]
    bs_results = data["bs results"]

    # acquire direct detection results
    bins = np.zeros(4)
    for trial in direct_results:
        bins += np.array(trial["counts"])
    bins *= (1 / sum(bins))

    print("bins:", bins)

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
        for trial in res:
            # We are looking at the cases where the entanglement was heralded by a detection at detector1
            counts.append(trial["counts1"])
            total += trial["total_count1"]
        counts = np.array(counts, dtype=float)
        # Here, the axis simply states that all 1st elements of sublists should be sum/std
        # together and all 2nd elements should be sum/std together. 
        st_dev = np.std(counts, axis=0)/(total/len(res)) 
        counts = np.sum(counts, axis=0)/total

        print("std:", st_dev, "mean:", counts)
        
        freq_0.append(counts[0])
        freq_1.append(counts[1])
        st_dev_0.append(st_dev[0])
        st_dev_1.append(st_dev[1])
    
    print("len of freq_0:", len(freq_0))

    # curve fitting
    params_0, _ = curve_fit(sin, phases, freq_0)
    params_1, _ = curve_fit(sin, phases, freq_1)
    vis_0 = abs(params_0[0]) / abs(params_0[2])
    vis_1 = abs(params_1[0]) / abs(params_1[2])

    # off-diagonal density matrix
    vis_total = (vis_0 + vis_1) / 2

    off_diag = vis_total * (bins[1] + bins[2]) / 2

    return bins, off_diag, freq_0, st_dev_0, freq_1, st_dev_1, params_0, params_1, vis_0, vis_1, vis_total

def build_bell_state(truncation, sign, phase=0, formalism="dm"):
    """Generate standard Bell state which is heralded in ideal BSM.

    For comparison with results from imperfect parameter choices.
    """

    basis0 = np.zeros(truncation+1)
    basis0[0] = 1
    basis1 = np.zeros(truncation+1)
    basis1[1] = 1
    basis10 = np.kron(basis1, basis0)
    basis01 = np.kron(basis0, basis1)
    
    if sign == "plus":
        ket = (basis10 + np.exp(1j*phase)*basis01)/np.sqrt(2)
    elif sign == "minus":
        ket = (basis10 - np.exp(1j*phase)*basis01)/np.sqrt(2)
    else:
        raise ValueError("Invalid Bell state sign type " + sign)

    dm = np.outer(ket, ket.conj())

    if formalism == "dm":
        return dm
    elif formalism == "ket":
        return ket
    else:
        raise ValueError("Invalid quantum state formalism " + formalism)
def effective_state(state):
    state_copy = copy(state)
    state_copy[0][0] = 0
    state_copy = state_copy/np.trace(state_copy)
    
    return state_copy
def calculate_fidelity(bins, off_diag):
    reconstructed_state = np.zeros((4,4), np.cdouble)
    for i in range(4):
        reconstructed_state[i,i] = bins[i]
    reconstructed_state[1,2] = reconstructed_state[2,1] = off_diag

    
    remaining_state_eff = effective_state(reconstructed_state)

    print("remaining_state_eff:\n", remaining_state_eff)

    # calculate the fidelity with reference Bell state
    bell_minus = build_bell_state(1, "plus")

    fidelity = np.trace(remaining_state_eff.dot(bell_minus)).real
    return fidelity

def plot_density_matrix(bins, off_diag, sin, directory, power):
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
    print("total:", total)
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
    plt.savefig(directory + f'density_{power}.png', bbox_inches='tight')
    plt.show()

def plot_interference(phases, freq_0, st_dev_0, freq_1, st_dev_1, params_0, params_1, directory, power):
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
    plt.savefig(directory + f'interference_{power}.png')
    plt.show()

directory = "results/masters4/final_data_O_COUNTER_4PAM/"

final_data_file = open(directory+"final_data.csv", 'w+')
final_data = writer(final_data_file)
final_data.writerow(["power", "diag00", "diag01", "diag10", "diag11", "visibility", "off_diag", "fidelity"])


for params_file_name in os.listdir(directory):
    # print("file:", params_file_name[:6])
    if params_file_name[:6] == "params":
        print("reading file", params_file_name)
        
        try:
            params_file = open(directory+params_file_name, "r")
            print(directory+params_file_name)
            params = load(params_file)
            power = params["AVG_POWER"]
        except:
            continue

        # #######REMOVE THIS#############
        # power = 10*np.log10(1000*power)

        if power == None:
            power = "No_Power"
        final_data_entry = [power]

        try:
            print("file name is:", f"absorptive{power}.json")
            data_file_name = directory + f"absorptive{power}.json"
            data = load(open(data_file_name))
        except:
            continue

        def sin(x, A, phi, z): return A*np.sin(x + phi) + z
        bins, off_diag, freq_0, st_dev_0, freq_1, st_dev_1, params_0, params_1, vis_0, vis_1, vis_total = analyze_data(data, sin)
        final_data_entry.extend(bins)
        final_data_entry.append(vis_total)
        final_data_entry.append(off_diag)

        plot_density_matrix(bins, off_diag, sin, directory, power)
        plot_interference(np.linspace(0, 2*np.pi, num=data["num_phase"]), freq_0, st_dev_0, freq_1, st_dev_1, params_0, params_1, directory, power)

        fidelity = calculate_fidelity(bins, off_diag)
        final_data_entry.append(fidelity)

        # output
        print("Detector 1 visibility:", vis_1)
        print("Detector 2 visibility:", vis_0)
        print("Combined:", vis_total)
        print("Off-diagonal:", off_diag)


        final_data.writerow(final_data_entry)

        # visibilities_file.write(f"\n{vis_total}")
final_data_file.close()
# visibilities_file.close()


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
