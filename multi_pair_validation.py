## IMPORT REQUIREMENTS

from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize
from multiprocessing import Process
from json import dump

from absorptive_experiment_misc.definitions import *
from src.kernel.timeline import Timeline
from src.kernel.quantum_manager import DENSITY_MATRIX_FORMALISM


## SIMULATION SETUP
# visibility at fringe bottom: .01 * .01 * .25/4
params = {
    "QUANTUM_WAVELENGTH" : 1536.,
    "SPDC_FREQUENCY" : 1e8,  # frequency of both SPDC sources' photon creation
    "MEAN_PHOTON_NUM" : 0.01,  # mean photon number of SPDC source on node 1
    "MPN_SETTINGS": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],

    # detectors
    "SIGNAL_DET_EFFICIENCY" : 0.5, # efficiency of detector 1 of BSM
    "IDLER_DET_EFFICIENCY" : 0.5,  # efficiency of detector 2 of BSM
    "SIGNAL_DET_DARK" : 0,
    "IDLER_DET_DARK" : 0,
    "TEMPORAL_COINCIDENCE_WINDOW": 400,
    "RESOLUTION": 20.,
    "SIGNAL_DET_DEAD" : 5000,
    "IDLER_DET_DEAD" : 5000,

    # fibers
    "SIGNAL_DIST" : 1.,  # distance between ANL and ERC, in km
    "IDLER_DIST" : 1.,  # distance between HC and ERC, in km
    "QUNATUM_ATTENUATION" : 0.,  # attenuation rate of optical fibre (in dB/km)
    'POLARIZATION_FIDELITY': 0.,

    # experiment settings
    "num_bs_trials_per_phase" : 1,
    "phase_settings" : [0, np.pi],

    "MODE_NUM": 10000000,

}

timeline = Timeline(1e12, formalism=DENSITY_MATRIX_FORMALISM)


## NETWORK SETUP

signal_receiver_name = "signal_receiver"
idler_receiver_name = "idler_receiver"
signal_receiver = PolarizationReceiverNode(signal_receiver_name, timeline, params)
idler_receiver = proxyReceiver(idler_receiver_name, timeline, signal_receiver)

source_node_name = "Polariation_source_node"
source_node = PolarizationDistributionNode(source_node_name, timeline, signal_receiver_name, idler_receiver_name, params)

qc_signal = add_quantum_channel(source_node, signal_receiver, timeline, distance = params["SIGNAL_DIST"], attenuation = params["QUNATUM_ATTENUATION"], density_matrix_tacking = True)
qc_idler = add_quantum_channel(source_node, idler_receiver, timeline, distance = params["IDLER_DIST"], attenuation = params["QUNATUM_ATTENUATION"], density_matrix_tacking = True)



## RUN SIMULATION AND ACQUISITION

coincidences = []
idler_singles = []
signal_singles = []

def run_simulations(idler_phase):
    print("\nNew Idler phase:", idler_phase)
    signal_receiver.reset()
    signal_receiver.rotateIdler(idler_phase)
    for signal_phase in params["phase_settings"]:
        timeline.init()
        print("New Signal phase:", signal_phase, "(Idler:", idler_phase, ")")
        signal_receiver.rotateSignal(signal_phase)
        source_node.start()
        timeline.run()

    new_signal_singles, new_idler_singles, new_coincidences = signal_receiver.get_data()
    # out_dict = {"signal":new_signal_singles, "idler":new_idler_singles, "coincidence":new_coincidences}
    # f = open(f"results/polarization/validation/outdata{idler_phase}.json", "w")
    # dump(out_dict, f)
    return new_coincidences[0], new_coincidences[1]


coinc_fringe_top = []
coinc_fringe_bottom = []

for mpn in params["MPN_SETTINGS"]:
    source_node.set_source_mpn(mpn)
    coinc_fringe_top_trial, coinc_fringe_bottom_trial = run_simulations(idler_phase=0)
    coinc_fringe_top.append(coinc_fringe_top_trial)
    coinc_fringe_bottom.append(coinc_fringe_bottom_trial)

exp_coinc_fringe_top = [params["MODE_NUM"] * (params["SIGNAL_DET_EFFICIENCY"]**2)*(mpn/2 + (mpn**2)/4) for mpn in params["MPN_SETTINGS"]]
exp_coinc_fringe_bottom = [params["MODE_NUM"] * (params["SIGNAL_DET_EFFICIENCY"]**2)*(mpn**2)/4 for mpn in ["MPN_SETTINGS"]]

plt.plot(coinc_fringe_top)
plt.plot(coinc_fringe_bottom)
plt.plot(exp_coinc_fringe_top)
plt.plot(exp_coinc_fringe_bottom)



# print("actual fringe top:", coinc_fringe_top, "actual fringe bottom:", coinc_fringe_bottom)
# print("expected fringe top:", exp_coinc_fringe_top, "expected fringe bottom:", exp_coinc_fringe_bottom)

# print("error in finge top:", abs(exp_coinc_fringe_top-coinc_fringe_top)/exp_coinc_fringe_top)
# print("error in finge bottom:", abs(exp_coinc_fringe_bottom-coinc_fringe_bottom)/exp_coinc_fringe_bottom)

# processes = [Process( target=run_simulations, args=(idler_phase,) ) for idler_phase in [0]]
# for p in processes:
#     p.start()

# for p in processes:
#     p.join()
    