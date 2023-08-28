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

if TYPE_CHECKING:
    from src.components.photon import Photon

from json import dump
import numpy as np

from src.kernel.event import Event
from src.kernel.process import Process
from src.kernel.timeline import Timeline
from src.kernel.quantum_manager import FOCK_DENSITY_MATRIX_FORMALISM
from src.components.detector import QSDetectorFockDirect, QSDetectorFockInterference
from src.components.light_source import SPDCSource
from src.components.memory import AbsorptiveMemory
from src.components.optical_channel import QuantumChannel, ClassicalChannel
from src.components.photon import Photon
from src.topology.node import Node
from src.protocol import Protocol
from src.kernel.quantum_utils import *  # only for manual calculation and should not be used in simulation


# define simulation constants

# quantum manager
TRUNCATION = 2  # truncation of Fock space (=dimension-1)

# photon sources
TELECOM_WAVELENGTH = 1436  # telecom band wavelength of SPDC source idler photon
WAVELENGTH = 606  # wavelength of AFC memory resonant absorption, of SPDC source signal photon
SPDC_FREQUENCY = 8e8  # frequency of both SPDC sources' photon creation (same as memory frequency and detector count rate)
MEAN_PHOTON_NUM1 = 0.1  # mean photon number of SPDC source on node 1
MEAN_PHOTON_NUM2 = 0.1  # mean photon number of SPDC source on node 2

# detectors
BSM_DET1_EFFICIENCY = 0.6  # efficiency of detector 1 of BSM
BSM_DET2_EFFICIENCY = 0.6  # efficiency of detector 2 of BSM
BSM_DET1_DARK = 0  # Dark count rate (Hz)
BSM_DET2_DARK = 0
MEAS_DET1_EFFICIENCY = 0.6  # efficiency of detector 1 of DM measurement
MEAS_DET2_EFFICIENCY = 0.6  # efficiency of detector 2 of DM measurement
MEAS_DET1_DARK = 0
MEAS_DET2_DARK = 0

# fibers
DIST_ANL_ERC = 20.  # distance between ANL and ERC, in m
DIST_HC_ERC = 20.  # distance between HC and ERC, in m
ATTENUATION = 0.2  # attenuation rate of optical fibre (in dB/km)
DELAY_CLASSICAL = 5e-3  # delay for classical communication between BSM node and memory nodes (in s)

# memories
MODE_NUM = 10000  # number of temporal modes of AFC memory (same for both memories)
MEMO_FREQUENCY1 = SPDC_FREQUENCY  # frequency of memory 1
MEMO_FREQUENCY2 = SPDC_FREQUENCY  # frequency of memory 2
ABS_EFFICIENCY1 = 0.35  # absorption efficiency of AFC memory 1
ABS_EFFICIENCY2 = 0.35  # absorption efficiency of AFC memory 2
PREPARE_TIME1 = 0  # time required for AFC structure preparation of memory 1
PREPARE_TIME2 = 0  # time required for AFC structure preparation of memory 2
COHERENCE_TIME1 = -1  # spin coherence time for AFC memory 1 spinwave storage, -1 means infinite time
COHERENCE_TIME2 = -1  # spin coherence time for AFC memory 2 spinwave storage, -1 means infinite time
AFC_LIFETIME1 = -1  # AFC structure lifetime of memory 1, -1 means infinite time
AFC_LIFETIME2 = -1  # AFC structure lifetime of memory 2, -1 means infinite time
DECAY_RATE1 = 4.3e-8  # retrieval efficiency decay rate for memory 1
DECAY_RATE2 = 4.3e-8  # retrieval efficiency decay rate for memory 2

# experiment settings
time = int(1e12)
calculate_fidelity_direct = True
calculate_rate_direct = True
num_direct_trials = 5
num_bs_trials_per_phase = 50
phase_settings = np.linspace(0, 2*np.pi, num=15, endpoint=False)

params = {
    # Detector_parameters
    "collection_probability" : BSM_DET1_EFFICIENCY,
    # "dark_count_rate" : BSM_DET1_DARK, #100,
    # "dead_time" : 25000,
    # "time_resolution" : 50,

    # Optical channel
    "quantum_channel_attenuation" : 0.44,
    "classical_channel_attenuation" : 0.55,
    "distance" : DIST_ANL_ERC, # Distance in km
    "raman_coefficient" : 10.5e-10, 
    # "max_rate" : 1e12,
    "quantum_channel_wavelength" : 1536e-9,
    "classical_channel_wavelength" : 1310e-9,
    "classical_communication_rate" : 1e10/1e12, # Classical communication (bit) rate in Picoseconds, i.e. B/ps. 1e7/1e12 is 10Mb/s in ps = 1e-5 b/ps

    "quantum_channel_index": 1.470,
    "classical_channel_index": 1.471,
    "classical_communication_window_size": (MODE_NUM/SPDC_FREQUENCY)*1e12,

    # Light Source
    # "mean_photon_num" : 0.003, # 0.01

    # Classical channel parameters
    "avg_power": 2e-3, # avg_power is written in W
    "OMA": 1, # OMA is written in dBm
    "narrow_band_filter_bandwidth" : 0.03,
}
OMA = 10**( params["OMA"] /10)/1000 # We receive the OMA in dBm
assert params["avg_power"] - OMA/2 > 0
params["classical_powers"] = [params["avg_power"]-OMA/2, params["avg_power"]-OMA/6, params["avg_power"]+OMA/6, params["avg_power"]+OMA/2]

# print("window size:", MODE_NUM/SPDC_FREQUENCY)


# function to generate standard pure Bell state for fidelity calculation
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
# retrieval efficiency as function of storage time for absorptive quantum memory, using exponential decay model
def efficiency1(t: int) -> float:
    return np.exp(-t*DECAY_RATE1)
def efficiency2(t: int) -> float:
    return np.exp(-t*DECAY_RATE2)
def add_quantum_channel(node1: Node, node2: Node, timeline: Timeline, **kwargs):
    name = "_".join(["qc", node1.name, node2.name])
    qc = QuantumChannel(name, timeline, **kwargs)
    qc.set_ends(node1, node2.name)
    return qc
def add_classical_channel(node1: Node, node2: Node, timeline: Timeline, **kwargs):
    name = "_".join(["cc", node1.name, node2.name])
    cc = ClassicalChannel(name, timeline, **kwargs)
    cc.set_ends(node1, node2.name)
    return cc


# Corrections: Include Batch size from the experimental_params dictionary.
# This is the equivalent of the raman_sender_protocol. Also, copy the classical message side of stuff 
# from the old function and include it ensure that the simulation can be stopped on time and processing started. 
class EmitProtocol(Protocol):
    def __init__(self, own: "EndNode", name: str, other_node: str, photon_pair_num: int,
                 source_name: str, memory_name: str):
        """Constructor for Emission protocol.

        Args:
            own (EndNode): node on which the protocol is located.
            name (str): name of the protocol instance.
            other_node (str): name of the other node to generate entanglement with
            photon_pair_num (int): number of output photon pulses to send in one execution.
            source_name (str): name of the light source on the node.
            memory_name (str): name of the memory on the node.
        """

        super().__init__(own, name)
        self.other_node = other_node
        self.num_output = photon_pair_num
        self.source_name = source_name
        self.memory_name = memory_name

    def start(self):
        if not self.own.components[self.memory_name].is_prepared:
            self.own.components[self.memory_name]._prepare_AFC()
        
        states = [None] * self.num_output
        # print("starting emission")
        self.own.components[self.source_name].emit(states)
        # print("done with inital emission")
    def received_message(self, src: str, msg):
        pass


# Corrections: Get rid of the get funtion. Also, make the source and memory send the qubits directly 
# to wherever it needs to be sent without invoking any application layer function. 
# This is the equivalent of the raman_sender in the previous implementation. 
class EndNode(Node):
    """Node for each end of the network (the memory node).

    This node stores an SPDC photon source and a quantum memory.
    The properties of attached devices are made customizable for each individual node.
    """

    def __init__(self, name: str, timeline: "Timeline", other_node: str, bsm_node: str, measure_node: str,
                 mean_photon_num: float, spdc_frequency: float, memo_frequency: float, abs_effi: float,
                 afc_efficiency: Callable, mode_number: int):
        super().__init__(name, timeline)

        self.bsm_name = bsm_node
        self.meas_name = measure_node

        # Initialize source and memory. 
        self.spdc_name = name + ".spdc_source"
        self.memo_name = name + ".memory"
        spdc = SPDCSource(self.spdc_name, timeline, wavelengths=[TELECOM_WAVELENGTH, WAVELENGTH],
                          frequency=spdc_frequency, mean_photon_num=mean_photon_num)
        memory = AbsorptiveMemory(self.memo_name, timeline, frequency=memo_frequency,
                                  absorption_efficiency=abs_effi, afc_efficiency=afc_efficiency,
                                  mode_number=mode_number, wavelength=WAVELENGTH, destination=measure_node)
        
        
        self.add_component(spdc)
        self.add_component(memory)
        spdc.add_receiver(self)
        spdc.add_receiver(memory)
        memory.add_receiver(self)

        # protocols
        self.emit_protocol = EmitProtocol(self, name + ".emit_protocol", other_node, mode_number, self.spdc_name, self.memo_name)

    # The light source always emits to the memory (take care of within the function) and to the 
    # BSM node for entanglement generation and heralding. Hence, if the dst is None, it has come from the 
    # light source and can be sent to the BSM node. 
    # The memory, however, can emit to the measure node, which can change. So, we must change the destnination in
    # the memory whenever we change the measure nodde. 
    def get(self, photon: "Photon", **kwargs):
        dst = kwargs.get("dst")
        if dst is None:
            # from spdc source: send to bsm node
            if not self.cchannels[self.bsm_name].classical_communication_running:
                print("classical dst is:", self.bsm_name)
                self.cchannels[self.bsm_name].start_classical_communication()
            self.send_qubit(self.bsm_name, photon)
        else:
            # from memory: send to destination (measurement) node
            if not self.cchannels[dst].classical_communication_running:
                print("classical dst is:", dst)
                self.cchannels[dst].start_classical_communication()
            self.send_qubit(dst, photon)


class EntangleNode(Node):
    def __init__(self, name: str, timeline: "Timeline", src_list: List[str]):
        super().__init__(name, timeline)

        # hardware setup
        self.bsm_name = name + ".bsm"
        # assume no relative phase between two input optical paths
        bsm = QSDetectorFockInterference(self.bsm_name, timeline, src_list)
        self.add_component(bsm)
        bsm.attach(self)
        self.set_first_component(self.bsm_name)
        self.resolution = max([d.time_resolution for d in bsm.detectors])

        # detector parameter setup
        bsm.set_detector(0, efficiency=BSM_DET1_EFFICIENCY, count_rate=SPDC_FREQUENCY, dark_count=BSM_DET1_DARK)
        bsm.set_detector(1, efficiency=BSM_DET2_EFFICIENCY, count_rate=SPDC_FREQUENCY, dark_count=BSM_DET1_DARK)

    def receive_qubit(self, src: str, qubit) -> None:
        self.components[self.first_component_name].get(qubit, src=src)

    def get_detector_entries(self, detector_name: str, start_time: int, num_bins: int, frequency: float):
        """Returns detection events for density matrix measurement. Used to determine BSM result.

        Args:
            detector_name (str): name of detector to get measurements from.
            start_time (int): simulation start time of when photons received.
            num_bins (int): number of arrival bins
            frequency (float): frequency of photon arrival (in Hz).

        Returns:
            List[int]: list of length (num_bins) with result for each time bin.
        """
        trigger_times = self.components[detector_name].get_photon_times()
        return_res = [0] * num_bins

        for time in trigger_times[0]:
            closest_bin = int(round((time - start_time) * frequency * 1e-12))
            expected_time = (float(closest_bin) * 1e12 / frequency) + start_time
            if abs(expected_time - time) < self.resolution and 0 <= closest_bin < num_bins:
                return_res[closest_bin] += 1

        for time in trigger_times[1]:
            closest_bin = int(round((time - start_time) * frequency * 1e-12))
            expected_time = (float(closest_bin) * 1e12 / frequency) + start_time
            if abs(expected_time - time) < self.resolution and 0 <= closest_bin < num_bins:
                return_res[closest_bin] += 2

        return return_res


class MeasureNode(Node):
    def __init__(self, name: str, timeline: "Timeline", other_nodes: List[str]):
        super().__init__(name, timeline)

        self.direct_detector_name = name + ".direct"
        direct_detector = QSDetectorFockDirect(self.direct_detector_name, timeline, other_nodes)
        self.add_component(direct_detector)
        direct_detector.attach(self)

        self.bs_detector_name = name + ".bs"
        bs_detector = QSDetectorFockInterference(self.bs_detector_name, timeline, other_nodes)
        self.add_component(bs_detector)
        bs_detector.add_receiver(self)

        self.set_first_component(self.direct_detector_name)

        # time resolution of SPDs
        self.resolution = max([d.time_resolution for d in direct_detector.detectors + bs_detector.detectors])

        # detector parameter setup
        direct_detector.set_detector(0, efficiency=MEAS_DET1_EFFICIENCY, count_rate=SPDC_FREQUENCY, dark_count=MEAS_DET1_DARK)
        direct_detector.set_detector(1, efficiency=MEAS_DET2_EFFICIENCY, count_rate=SPDC_FREQUENCY, dark_count=MEAS_DET2_DARK)
        bs_detector.set_detector(0, efficiency=MEAS_DET1_EFFICIENCY, count_rate=SPDC_FREQUENCY, dark_count=MEAS_DET1_DARK)
        bs_detector.set_detector(1, efficiency=MEAS_DET2_EFFICIENCY, count_rate=SPDC_FREQUENCY, dark_count=MEAS_DET2_DARK)

    def receive_qubit(self, src: str, qubit) -> None:
        self.components[self.first_component_name].get(qubit, src=src)

    def set_phase(self, phase: float):
        self.components[self.bs_detector_name].set_phase(phase)

    def get_detector_entries(self, detector_name: str, start_time: int, num_bins: int, frequency: float):
        """Returns detection events for density matrix measurement.

        Args:
            detector_name (str): name of detector to get measurements from.
            start_time (int): simulation start time of when photons received.
            num_bins (int): number of arrival bins
            frequency (float): frequency of photon arrival (in Hz).

        Returns:
            List[int]: list of length (num_bins) with result for each time bin.
        """
        print("start time is:", start_time, "bin width:", 1/(frequency*1e12))
        trigger_times = self.components[detector_name].get_photon_times()
        print("len of trigger times (Measure node):", len(trigger_times))
        return_res = [0] * num_bins

        for time in trigger_times[0]:
            closest_bin = int(round((time - start_time) * frequency * 1e-12))
            expected_time = (float(closest_bin) * 1e12 / frequency) + start_time
            # print("closest bin:", closest_bin)
            
            if abs(expected_time - time) < self.resolution and 0 <= closest_bin < num_bins:
                return_res[closest_bin] += 1
            else:
                print("actual_time:", time, "expected_time", expected_time, "error:", abs(expected_time - time), "closest_bin:", closest_bin )
                # print("too far gone, closest bin:", closest_bin)

        for time in trigger_times[1]:
            closest_bin = int(round((time - start_time) * frequency * 1e-12))
            expected_time = (float(closest_bin) * 1e12 / frequency) + start_time
            if abs(expected_time - time) < self.resolution and 0 <= closest_bin < num_bins:
                return_res[closest_bin] += 2
            else:
                print("actual_time:", time, "expected_time", expected_time, "error:", abs(expected_time - time), "closest_bin:", closest_bin )
            
        return return_res


if __name__ == "__main__":

    tl = Timeline(time, formalism=FOCK_DENSITY_MATRIX_FORMALISM, truncation=TRUNCATION)

    anl_name = "Argonne0"
    hc_name = "Harper Court1"
    erc_name = "Eckhardt Research Center BSM2"
    erc_2_name = "Eckhardt Research Center Measurement3"
    seeds = [1, 2, 3, 4]
    src_list = [anl_name, hc_name]  # the list of sources, note the order

    anl = EndNode(anl_name, tl, hc_name, erc_name, erc_2_name, mean_photon_num=MEAN_PHOTON_NUM1,
                  spdc_frequency=SPDC_FREQUENCY, memo_frequency=MEMO_FREQUENCY1, abs_effi=ABS_EFFICIENCY1,
                  afc_efficiency=efficiency1, mode_number=MODE_NUM)
    hc = EndNode(hc_name, tl, anl_name, erc_name, erc_2_name, mean_photon_num=MEAN_PHOTON_NUM2,
                 spdc_frequency=SPDC_FREQUENCY, memo_frequency=MEMO_FREQUENCY2, abs_effi=ABS_EFFICIENCY2,
                 afc_efficiency=efficiency2, mode_number=MODE_NUM)
    erc = EntangleNode(erc_name, tl, src_list)
    erc_2 = MeasureNode(erc_2_name, tl, src_list)

    for seed, node in zip(seeds, [anl, hc, erc, erc_2]):
        node.set_seed(seed)

    # extend fiber lengths to be equivalent
    fiber_length = max(DIST_ANL_ERC, DIST_HC_ERC)

    qc1 = add_quantum_channel(anl, erc, tl, attenuation=ATTENUATION, distance=fiber_length)
    qc2 = add_quantum_channel(hc, erc, tl, attenuation=ATTENUATION, distance=fiber_length)
    qc3 = add_quantum_channel(anl, erc_2, tl, attenuation=ATTENUATION, distance=fiber_length)
    qc4 = add_quantum_channel(hc, erc_2, tl, attenuation=ATTENUATION, distance=fiber_length)

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
    print("start_time_meas:", start_time_meas)

    # Start
    params["simulation_end_time"] = start_time_meas
    # print("simulation tim:", params["simulation_end_time"])


    results_direct_measurement = []
    results_bs_measurement = [[] for _ in phase_settings]

    """Run Simulation"""
    # for i in range(num_direct_trials):
    #     # start protocol for emitting
    #     process = Process(anl.emit_protocol, "start", [])
    #     event = Event(start_time_anl, process)
    #     tl.schedule(event)
    #     process = Process(hc.emit_protocol, "start", [])
    #     event = Event(start_time_hc, process)
    #     tl.schedule(event)

    #     tl.run()
    #     print("finished direct measurement trial {} out of {}".format(i+1, num_direct_trials))

    #     # collect data

    #     # BSM results determine relative sign of reference Bell state and herald successful entanglement
    #     bsm_res = erc.get_detector_entries(erc.bsm_name, start_time_bsm, MODE_NUM, SPDC_FREQUENCY)
    #     bsm_success_indices = [i for i, res in enumerate(bsm_res) if res == 1 or res == 2]
    #     meas_res = erc_2.get_detector_entries(erc_2.direct_detector_name, start_time_meas, MODE_NUM, SPDC_FREQUENCY)

    #     num_bsm_res = len(bsm_success_indices)
    #     meas_res_valid = [meas_res[i] for i in bsm_success_indices]
    #     counts_diag = [0] * 4
    #     for j in range(4):
    #         counts_diag[j] = meas_res_valid.count(j)
    #     res_diag = {"counts": counts_diag, "total_count": num_bsm_res}
    #     results_direct_measurement.append(res_diag)

    #     # reset timeline
    #     tl.time = 0
    #     tl.init()

    # change to other measurement
    erc_2.set_first_component(erc_2.bs_detector_name)
    
    
    # classical_omas = np.linspace(-1,3,5)
    avg_powers = np.linspace(-1,4,5)
    avg_powers = 10**( avg_powers /10)/1000
    for power in avg_powers: 
        OMA = 10**( params["OMA"] /10)/1000 # We receive the OMA in dBm
        assert params["avg_power"] - OMA/2 > 0

        # We have the new parameters here. Calculate the BER for the classical communication. 
        # Use that BER to feed into NS3's call using command line arguments. 
        # Inside NS3, prepare the PCAP files. Then, let the simulator take over, read the pcap files
        # and perform the simulation. 

        params["classical_powers"] = [power-OMA/2, power-OMA/6, power+OMA/6, power+OMA/2]
        
        for i, phase in enumerate(phase_settings):
            print()
            print("New Phase angle")
            erc_2.set_phase(phase)

            for j in range(num_bs_trials_per_phase):
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
                    j+1, num_bs_trials_per_phase, i+1, len(phase_settings)))


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
        filename = f"results/all_Raman_long_450_power_many_photons/absorptive{power}.json"
        fh = open(filename, 'w')
        info = {"num_direct_trials": num_direct_trials, "num_bs_trials": num_bs_trials_per_phase,
                "num_phase": len(phase_settings),
                "direct results": results_direct_measurement, "bs results": results_bs_measurement}
        dump(info, fh)
