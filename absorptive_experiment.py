"""The main script for simulating experiment of entanglement generation between two remote AFC absorptive quantum memories.

There are 4 nodes involved: 2 memory nodes, 1 entangling node (for BSM) and 1 measurement node (for measurement of retrieved photonic state).
Each memory node is connected with both entangling node and measurement node, but there is no direct connection between memory nodes.

Each memory node contains an AFC memory instance and an SPDC source instance.
The entangling node contains a QSDetectorFockInterference instance (BSM device with a beamsplitter and two photon detectors behind).
The measurement node contians a QSDetectorFockDirect instance and a QSDetectorFockInterference instance, for measurement of 
    diagonal and off-diagonal elements of the effective 4-d density matrix, respectively.
"""

from typing import List, Callable, TYPE_CHECKING
from pathlib import Path
from copy import copy
import json

if TYPE_CHECKING:
    from src.components.photon import Photon

from json import dump
import numpy as np

from absorptive_experiment_misc.definitions import *

from src.kernel.event import Event
from src.kernel.process import Process
from src.kernel.timeline import Timeline
from src.kernel.quantum_manager import FOCK_DENSITY_MATRIX_FORMALISM
from src.topology.node import Node
from src.kernel.quantum_utils import *  # only for manual calculation and should not be used in simulation
from src.classical_communication.BER_models import calculate_BER_4PAM_noPreamplifier
from src.classical_communication.runner_ns3 import run_NS3

# sim_params_file = open("params.json")
# sim_params = json.loads(sim_params_file)


# global parameters
SPDC_FREQUENCY = 8e8
MODE_NUM = 100

# Note that all classical channel dependent quantities: 
#      classical_powers
#      raman_coefficient
#      classical_channel_attenuation
#      classical_channel_index
# Have been written in the fashion: [1310nm (O-band), 1610nm (L-band)]

params = {
    # quantum manager
    "TRUNCATION" : 2,  # truncation of Fock space (:dimension-1)

    # photon sources
    "TELECOM_WAVELENGTH" : 1436.,  # telecom band wavelength of SPDC source idler photon
    "QUANTUM_WAVELENGTH" : 1536.,  # wavelength of AFC memory resonant absorption, of SPDC source signal photon
    "SPDC_FREQUENCY" : SPDC_FREQUENCY,  # frequency of both SPDC sources' photon creation (same as memory frequency and detector count rate)
    "MEAN_PHOTON_NUM1" : 0.1,  # mean photon number of SPDC source on node 1
    "MEAN_PHOTON_NUM2" : 0.1,  # mean photon number of SPDC source on node 2


    # detectors
    "BSM_DET1_EFFICIENCY" : 0.6, # efficiency of detector 1 of BSM
    "BSM_DET2_EFFICIENCY" : 0.6,  # efficiency of detector 2 of BSM
    "BSM_DET1_DARK" : 0,  # Dark count rate (Hz)
    "BSM_DET2_DARK" : 0,
    "MEAS_DET1_EFFICIENCY" : 0.6,  # efficiency of detector 1 of DM measurement
    "MEAS_DET2_EFFICIENCY" : 0.6,  # efficiency of detector 2 of DM measurement
    "MEAS_DET1_DARK" : 0,
    "MEAS_DET2_DARK" : 0,
    "NBF_BANDWIDTH" : 0.03,

    # fibers
    "DIST_ANL_ERC" : 20.,  # distance between ANL and ERC, in m
    "DIST_HC_ERC" : 20.,  # distance between HC and ERC, in m
    "QUNATUM_ATTENUATION" : 0.2,  # attenuation rate of optical fibre (in dB/km)
    "CLASSICAL_ATTENUATION" : [0.55, 0.5],
    "RAMAN_COEFFICIENTS" : [10.5e-10, 33.e-10],
    "DELAY_CLASSICAL" : 5e-3,  # delay for classical communication between BSM node and memory nodes (in s)
    "QUANTUM_INDEX" : 1.470,
    "CLASSICAL_INDEX" : [1.471, 1.470],

    # classical communication 
    "CLASSICAL_WAVELENGTH" : 1310,
    "CLASSICAL_RATE" : 1e10/1e12,
    "COMMS_WINDOW_SIZE" : (MODE_NUM/SPDC_FREQUENCY)*1e12, # Amount of time it takes to complete one batch of MODE_NUM quantum emissions. 
                                                          # This is the amount of time that we calculate Raman scatting for. 
    "AVG_POWER": 2e-3,
    "OMA": 1,

    # memories
    "MODE_NUM" : MODE_NUM,  # number of temporal modes of AFC memory (same for both memories)
    "MEMO_FREQUENCY1" : SPDC_FREQUENCY,  # frequency of memory 1
    "MEMO_FREQUENCY2" : SPDC_FREQUENCY,  # frequency of memory 2
    "ABS_EFFICIENCY1" : 0.35,  # absorption efficiency of AFC memory 1
    "ABS_EFFICIENCY2" : 0.35,  # absorption efficiency of AFC memory 2
    "PREPARE_TIME1" : 0,  # time required for AFC structure preparation of memory 1
    "PREPARE_TIME2" : 0, # time required for AFC structure preparation of memory 2
    "COHERENCE_TIME1" : -1,  # spin coherence time for AFC memory 1 spinwave storage, -1 means infinite time
    "COHERENCE_TIME2" : -1,  # spin coherence time for AFC memory 2 spinwave storage, -1 means infinite time
    "AFC_LIFETIME1" : -1,  # AFC structure lifetime of memory 1, -1 means infinite time
    "AFC_LIFETIME2" : -1,  # AFC structure lifetime of memory 2, -1 means infinite time
    "DECAY_RATE1" : 4.3e-8,  # retrieval efficiency decay rate for memory 1
    "DECAY_RATE2" : 4.3e-8,  # retrieval efficiency decay rate for memory 2

    # experiment settings
    "time" : int(1e12),
    "calculate_fidelity_direct" : True,
    "calculate_rate_direct" : True,
    "num_direct_trials" : 1,
    "num_bs_trials_per_phase" : 1,
    "phase_settings" : list(np.linspace(0, 2*np.pi, num=1, endpoint=False)),
}

OMA = 10**( params["OMA"] /10)/1000 # We receive the OMA in dBm
assert params["AVG_POWER"] - OMA/2 > 0
params["CLASSICAL_POWERS"] = [[params["AVG_POWER"]-OMA/2, params["AVG_POWER"]-OMA/6, params["AVG_POWER"]+OMA/6, params["AVG_POWER"]+OMA/2],
                              [params["AVG_POWER"]-OMA/2, params["AVG_POWER"]-OMA/6, params["AVG_POWER"]+OMA/6, params["AVG_POWER"]+OMA/2]]


def afc_efficiency1(t: int) -> float:
    return np.exp(-t*params["DECAY_RATE1"])
def afc_efficiency2(t: int) -> float:
    return np.exp(-t*params["DECAY_RATE2"])


if __name__ == "__main__":

    tl = Timeline(params["time"], formalism=FOCK_DENSITY_MATRIX_FORMALISM, truncation=params["TRUNCATION"])

    anl_name = "Argonne0"
    hc_name = "Harper Court1"
    erc_name = "Eckhardt Research Center BSM2"
    erc_2_name = "Eckhardt Research Center Measurement3"
    seeds = [1, 2, 3, 4]
    src_list = [anl_name, hc_name]  # the list of sources, note the order

    anl = EndNode(anl_name, tl, hc_name, erc_name, erc_2_name, mean_photon_num=params["MEAN_PHOTON_NUM1"],
                  spdc_frequency=SPDC_FREQUENCY, memo_frequency=params["MEMO_FREQUENCY1"], abs_effi=params["ABS_EFFICIENCY1"],
                  afc_efficiency=afc_efficiency1, mode_number=MODE_NUM, TELECOM_WAVELENGTH = params["TELECOM_WAVELENGTH"], WAVELENGTH = params["QUANTUM_WAVELENGTH"])
    hc = EndNode(hc_name, tl, anl_name, erc_name, erc_2_name, mean_photon_num=params["MEAN_PHOTON_NUM2"],
                 spdc_frequency=SPDC_FREQUENCY, memo_frequency=params["MEMO_FREQUENCY2"], abs_effi=params["ABS_EFFICIENCY2"],
                 afc_efficiency=afc_efficiency2, mode_number=MODE_NUM, TELECOM_WAVELENGTH = params["TELECOM_WAVELENGTH"], WAVELENGTH = params["QUANTUM_WAVELENGTH"])
    erc = EntangleNode(erc_name, tl, src_list, params["BSM_DET1_EFFICIENCY"], params["BSM_DET2_EFFICIENCY"], params["SPDC_FREQUENCY"], params["BSM_DET1_DARK"])
    erc_2 = MeasureNode(erc_2_name, tl, src_list, params["MEAS_DET1_EFFICIENCY"], params["MEAS_DET2_EFFICIENCY"], params["SPDC_FREQUENCY"], params["BSM_DET1_DARK"], params["MEAS_DET2_DARK"])

    for seed, node in zip(seeds, [anl, hc, erc, erc_2]):
        node.set_seed(seed)

    # extend fiber lengths to be equivalent
    fiber_length = max(params["DIST_ANL_ERC"], params["DIST_HC_ERC"])

    qc1 = add_quantum_channel(anl, erc, tl, attenuation=params["QUNATUM_ATTENUATION"], distance=fiber_length)
    qc2 = add_quantum_channel(hc, erc, tl, attenuation=params["QUNATUM_ATTENUATION"], distance=fiber_length)
    qc3 = add_quantum_channel(anl, erc_2, tl, attenuation=params["QUNATUM_ATTENUATION"], distance=fiber_length)
    qc4 = add_quantum_channel(hc, erc_2, tl, attenuation=params["QUNATUM_ATTENUATION"], distance=fiber_length)

    cc1 = add_classical_channel(anl, erc, tl, distance=fiber_length, params = params)
    cc2 = add_classical_channel(hc, erc, tl, distance=fiber_length, params = params)
    cc3 = add_classical_channel(anl, erc_2, tl, distance=fiber_length, params = params)
    cc4 = add_classical_channel(hc, erc_2, tl, distance=fiber_length, params = params)

    tl.init()

    # calculate start time for protocol
    # since fiber lengths equal, both start at 0
    start_time_anl = start_time_hc = 0

    # calculations for when to start recording measurements
    delay_anl = anl.qchannels[erc_2_name].delay
    delay_hc = hc.qchannels[erc_2_name].delay
    assert delay_anl == delay_hc
    start_time_bsm = start_time_anl + delay_anl
    mem = anl.components[anl.memo_name]
    # total_time is the total amount of time that 
    total_time = mem.total_time
    start_time_meas = start_time_anl + total_time + delay_anl
    # print("start_time_meas:", start_time_meas)

    # Start
    params["simulation_end_time"] = start_time_meas
    # print("simulation tim:", params["simulation_end_time"])


    results_direct_measurement = []
    results_bs_measurement = [[] for _ in params["phase_settings"]]

    NS3_file = "src/classical_communication/ns3_script.cc"
    data_file = "src/classical_communication/file.jpg"

    """Run Simulation"""
    # change to other measurement
    erc_2.set_first_component(erc_2.bs_detector_name)
    
    
    # classical_omas = np.linspace(-1,3,5)
    # avg_powers = np.linspace(-1,4,5)
    avg_powers = np.array([3])
    avg_powers = 10**( avg_powers /10)/1000
    params_file = open("results/exp_large_mode_num/params.json", "w+")
    json.dump(params, params_file)
    for power in avg_powers: 
        OMA = 10**( params["OMA"] /10)/1000 # We receive the OMA in dBm
        assert params["AVG_POWER"] - OMA/2 > 0
        params["CLASSICAL_POWERS"] = [[power-OMA/2, power-OMA/6, power+OMA/6, power+OMA/2], [power-OMA/2, power-OMA/6, power+OMA/6, power+OMA/2]]

        print("params[CLASSICAL_POWERS][0]:", params["CLASSICAL_POWERS"])
        BER = calculate_BER_4PAM_noPreamplifier(params["CLASSICAL_ATTENUATION"][0], params["DIST_ANL_ERC"], params["CLASSICAL_RATE"]*1e12, params["CLASSICAL_POWERS"][0], params["AVG_POWER"])
        run_NS3(BER, NS3_file, data_file, data_file)

        # We have the new parameters here. Calculate the BER for the classical communication. 
        # Use that BER to feed into NS3's call using command line arguments. 
        # Inside NS3, prepare the PCAP files. Then, let the simulator take over, read the pcap files
        # and perform the simulation. 
        
        for i, phase in enumerate(params["phase_settings"]):
            print()
            print("New Phase angle")
            erc_2.set_phase(phase)

            for j in range(params["num_bs_trials_per_phase"]):
                print()
                print("New Trial")
                # start protocol for emitting
                # cc1.start_classical_communication()
                # cc2.start_classical_communication()
                # cc3.start_classical_communication()
                # cc4.start_classical_communication()

                process = Process(anl.emit_protocol, "start", [])
                event = Event(start_time_anl, process)
                tl.schedule(event)



                process = Process(hc.emit_protocol, "start", [])
                event = Event(start_time_hc, process)
                tl.schedule(event)

                tl.run()
                print("finished interference measurement trial {} out of {} for phase {} out ouf {}".format(
                    j+1, params["num_bs_trials_per_phase"], i+1, len(params["phase_settings"])))


                ########## Resetting classical communication #################
                cc3.classical_communication_running = False
                cc4.classical_communication_running = False
                cc1.classical_communication_running = False
                cc2.classical_communication_running = False

                # collect data

                # relative sign should influence interference pattern
                bsm_res = erc.get_detector_entries(erc.bsm_name, start_time_bsm, MODE_NUM, SPDC_FREQUENCY)
                bsm_success_indices_1 = [i for i, res in enumerate(bsm_res) if res == 1]
                bsm_success_indices_2 = [i for i, res in enumerate(bsm_res) if res == 2]
                print("entangling measurment results:", bsm_res)
                meas_res = erc_2.get_detector_entries(erc_2.bs_detector_name, start_time_meas, MODE_NUM, SPDC_FREQUENCY)
                res_interference = {}
                print("measuring measurment results:", meas_res)

                print()

                # detector 1
                num_bsm_res = len(bsm_success_indices_1)
                meas_res_valid = [meas_res[i] for i in bsm_success_indices_1]
                num_detector_0 = meas_res_valid.count(1) + meas_res_valid.count(3)
                num_detector_1 = meas_res_valid.count(2) + meas_res_valid.count(3)
                counts_interfere = [num_detector_0, num_detector_1]
                res_interference["counts1"] = counts_interfere
                res_interference["total_count1"] = num_bsm_res

                # detector 2
                num_bsm_res = len(bsm_success_indices_2)
                meas_res_valid = [meas_res[i] for i in bsm_success_indices_2]
                num_detector_0 = meas_res_valid.count(1) + meas_res_valid.count(3)
                num_detector_1 = meas_res_valid.count(2) + meas_res_valid.count(3)
                counts_interfere = [num_detector_0, num_detector_1]
                res_interference["counts2"] = counts_interfere
                res_interference["total_count2"] = num_bsm_res

                results_bs_measurement[i].append(res_interference)

                # reset timeline
                tl.time = 0
                tl.init()

        """Store results"""

        # open file to store experiment results
        Path("results").mkdir(parents=True, exist_ok=True)
        filename = f"results/try/absorptive{power}.json"
        fh = open(filename, 'w')
        info = {"num_direct_trials": params["num_direct_trials"], "num_bs_trials": params["num_bs_trials_per_phase"],
                "num_phase": len(params["phase_settings"]),
                "direct results": results_direct_measurement, "bs results": results_bs_measurement}
        dump(info, fh)
