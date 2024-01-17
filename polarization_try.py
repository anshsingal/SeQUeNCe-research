# from typing import List, Callable, TYPE_CHECKING
from pathlib import Path
from copy import copy
from matplotlib import pyplot as plt

# if TYPE_CHECKING:
#     from src.components.photon import Photon

import numpy as np

from absorptive_experiment_misc.definitions import *

from src.kernel.event import Event
from src.kernel.process import Process
from src.kernel.timeline import Timeline
from src.kernel.quantum_manager import KET_STATE_FORMALISM, DENSITY_MATRIX_FORMALISM

params = {
    "QUANTUM_WAVELENGTH" : 1536.,  # wavelength of AFC memory resonant absorption, of SPDC source signal photon
    "SPDC_FREQUENCY" : 1e8,  # frequency of both SPDC sources' photon creation
    "MEAN_PHOTON_NUM" : 0.7,  # mean photon number of SPDC source on node 1

    # detectors
    "SIGNAL_DET_EFFICIENCY" : 0.15, # efficiency of detector 1 of BSM
    "IDLER_DET_EFFICIENCY" : 0.15,  # efficiency of detector 2 of BSM
    "SIGNAL_DET_DARK" : 0,
    "IDLER_DET_DARK" : 0,
    "TEMPORAL_COINCIDENCE_WINDOW": 400,
    "RESOLUTION": 20.,
    "SIGNAL_DET_DEAD" : 5000,
    "IDLER_DET_DEAD" : 5000,

    # fibers
    "SIGNAL_DIST" : 10.,  # distance between ANL and ERC, in km
    "IDLER_DIST" : 10.,  # distance between HC and ERC, in km
    "QUNATUM_ATTENUATION" : 0.076  * 10/np.log(10),  # attenuation rate of optical fibre (in dB/km)
    "QUANTUM_INDEX" : 1.471,

    # experiment settings
    "num_bs_trials_per_phase" : 200,
    "phase_settings" : (np.linspace(0, 2*np.pi, num=50, endpoint=False)),

    "MODE_NUM": 10,
}

timeline = Timeline(1e12, formalism=DENSITY_MATRIX_FORMALISM)

signal_receiver_name = "signal_receiver"
idler_receiver_name = "idler_receiver"
signal_receiver = PolarizationReceiverNode(signal_receiver_name, timeline, params)
idler_receiver = proxyReceiver(idler_receiver_name, timeline, signal_receiver)

source_node_name = "Polariation_source_node"
source_node = PolarizationDistributionNode(source_node_name, timeline, signal_receiver_name, idler_receiver_name, params)

qc_signal = add_quantum_channel(source_node, signal_receiver, timeline, distance = params["SIGNAL_DIST"], attenuation = params["QUNATUM_ATTENUATION"], density_matrix_tacking = True)
qc_idler = add_quantum_channel(source_node, idler_receiver, timeline, distance = params["IDLER_DIST"], attenuation = params["QUNATUM_ATTENUATION"], density_matrix_tacking = True)

coincidences = []
det_idler_singles = []
det_signal_singles = []

for i, phase in enumerate(params["phase_settings"]):
    timeline.init()
    print("\nNew Phase angle:", phase)
    signal_receiver.rotate(phase)

    for j in range(params["num_bs_trials_per_phase"]):
        source_node.start()
        timeline.run()
    # print("signal_counts", signal_receiver.detections[signal_receiver.signal_detector])
    # print("idler_counts", signal_receiver.detections[signal_receiver.idler_detector])

    coincidences.append(signal_receiver.coincidence_count)
    det_idler_singles.append(signal_receiver.det_idler_singles_count+signal_receiver.coincidence_count)
    signal_receiver.det_idler_singles_count = 0
    det_signal_singles.append(signal_receiver.det_signal_singles_count+signal_receiver.coincidence_count)
    signal_receiver.det_signal_singles_count = 0
    signal_receiver.coincidence_count = 0

signal_singles, idler_singles, coincidences = signal_receiver.get_data()
print("signal_singles:", signal_singles, len(signal_singles))
print("idler_singles:", idler_singles, len(idler_singles))
print("coincidences:", coincidences, len(coincidences))


plt.figure()
plt.ylim([0, max(coincidences)])
# print("len of phases", len(params["num_bs_trials_per_phase"]))
plt.plot(params["phase_settings"], coincidences, label = "Hvis")

plt.title("Coincidences")
plt.legend()
plt.figure()
plt.ylim([0, max(det_idler_singles)])
plt.plot(params["phase_settings"], det_idler_singles, 'x')
plt.title("Idler Detector singles counts")
plt.figure()
plt.ylim([0, max(det_signal_singles)])
plt.plot(params["phase_settings"], det_signal_singles)
plt.title("Signal detector singles counts")

plt.show()