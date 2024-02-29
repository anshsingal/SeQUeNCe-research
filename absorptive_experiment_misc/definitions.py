from typing import Callable
import numpy as np
import random

from src.kernel.timeline import Timeline
from src.components.detector import QSDetectorFockDirect, QSDetectorFockInterference, Detector
from src.components.light_source import SPDCSource
from src.components.memory import AbsorptiveMemory
from src.components.photon import Photon
from src.components.polarizer import Polarizer
from src.topology.node import Node
from src.protocol import Protocol
from src.kernel.quantum_utils import *  # only for manual calculation and should not be used in simulation
from src.components.optical_channel import QuantumChannel, ClassicalChannel
from src.kernel.quantum_state import FreeQuantumState
from src.utils.encoding import polarization

# retrieval efficiency as function of storage time for absorptive quantum memory, using exponential decay model
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


class EndNode(Node):
    """Node for each end of the network (the memory node).

    This node stores an SPDC photon source and a quantum memory.
    The properties of attached devices are made customizable for each individual node.
    """

    def __init__(self, name: str, timeline: "Timeline", other_node: str, bsm_node: str, measure_node: str,
                 mean_photon_num: float, spdc_frequency: float, memo_frequency: float, abs_effi: float,
                 afc_efficiency: Callable, mode_number: int, TELECOM_WAVELENGTH, WAVELENGTH):
        super().__init__(name, timeline)

        self.bsm_name = bsm_node
        self.meas_name = measure_node
        self.start_classical_communication = False

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
            if self.start_classical_communication and not self.cchannels[self.bsm_name].classical_communication_running:
                # print("classical dst is:", self.bsm_name)
                self.cchannels[self.bsm_name].start_classical_communication()
            self.send_qubit(self.bsm_name, photon)
        else:
            # from memory: send to destination (measurement) node
            # if not self.cchannels[dst].classical_communication_running:
                # print("classical dst is:", dst)
                # self.cchannels[dst].start_classical_communication()
            self.send_qubit(dst, photon)


class EntangleNode(Node):
    def __init__(self, name: str, timeline: "Timeline", src_list: List[str], BSM_DET1_EFFICIENCY, BSM_DET2_EFFICIENCY, SPDC_FREQUENCY, BSM_DET1_DARK):
        super().__init__(name, timeline)

        # hardware setup
        self.bsm_name = name + ".bsm"
        # assume no relative phase between two input optical paths
        bsm = QSDetectorFockInterference(self.bsm_name, timeline, src_list)
        self.add_component(bsm)
        bsm.attach(self)
        self.set_first_component(self.bsm_name)
        self.temporal_coincidence_window = max([d.temporal_coincidence_window for d in bsm.detectors])

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

        # The detector_name is a misleading title. it is actually the BSM name. We are getting detections from both detectors.
        trigger_times = self.components[detector_name].get_photon_times()
        # print("trigger_times:", trigger_times)

        # print("len of trigger times (BSM node):", len(trigger_times[0]), len(trigger_times[1]))

        return_res = [0] * num_bins

        num_rejects_0 = 0
        num_rejects_1 = 0

        for time in trigger_times[0]:
            closest_bin = int(round((time - start_time) * frequency * 1e-12))
            expected_time = (float(closest_bin) * 1e12 / frequency) + start_time
            if abs(expected_time - time) < self.temporal_coincidence_window and 0 <= closest_bin < num_bins:
                return_res[closest_bin] += 1
                # print("1 at", closest_bin)
            else:
                # print("time diff:", abs(expected_time - time), "closest_bin:", closest_bin)
                num_rejects_0 += 1

        for time in trigger_times[1]:
            closest_bin = int(round((time - start_time) * frequency * 1e-12))
            expected_time = (float(closest_bin) * 1e12 / frequency) + start_time
            if abs(expected_time - time) < self.temporal_coincidence_window and 0 <= closest_bin < num_bins:
                return_res[closest_bin] += 2
                # print("2 at", closest_bin)
            else:
                # print("time diff:", abs(expected_time - time), "closest_bin:", closest_bin)
                num_rejects_1 += 1

        print("len of trigger times (BSM node):", len(trigger_times[0]), len(trigger_times[1]), "BSM num_rejects:", num_rejects_0, num_rejects_1)

        return return_res


class MeasureNode(Node):
    def __init__(self, name: str, timeline: "Timeline", other_nodes: List[str], MEAS_DET1_EFFICIENCY, MEAS_DET2_EFFICIENCY, SPDC_FREQUENCY, MEAS_DET1_DARK, MEAS_DET2_DARK):
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
        self.temporal_coincidence_window = max([d.temporal_coincidence_window for d in direct_detector.detectors + bs_detector.detectors])

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
        # print("start time is:", start_time, "bin width:", 1/(frequency*1e12))
        trigger_times = self.components[detector_name].get_photon_times()
        # print("trigger_times:", trigger_times)
        # print("len of trigger times (Measure node):", len(trigger_times[0]), len(trigger_times[1]))
        return_res = [0] * num_bins

        # print("num of detector triggers:", len(trigger_times[0]), len(trigger_times[1]))

        num_rejects_0 = 0
        num_rejects_1 = 0


        for time in trigger_times[0]:
            closest_bin = int(round((time - start_time) * frequency * 1e-12))
            expected_time = (float(closest_bin) * 1e12 / frequency) + start_time
            # print("closest bin:", closest_bin)
            
            if abs(expected_time - time) < self.temporal_coincidence_window and 0 <= closest_bin < num_bins:
                return_res[closest_bin] += 1
                # print("1 at", closest_bin)
            else:
                num_rejects_0 += 1
                # print("actual_time:", time, "expected_time", expected_time, "error:", abs(expected_time - time), "closest_bin:", closest_bin )
                # print("too far gone, closest bin:", closest_bin)


        for time in trigger_times[1]:
            closest_bin = int(round((time - start_time) * frequency * 1e-12))
            expected_time = (float(closest_bin) * 1e12 / frequency) + start_time
            if abs(expected_time - time) < self.temporal_coincidence_window and 0 <= closest_bin < num_bins:
                return_res[closest_bin] += 2
                # print("2 at", closest_bin)
            else:
                num_rejects_1 += 1
                # print("actual_time:", time, "expected_time", expected_time, "error:", abs(expected_time - time), "closest_bin:", closest_bin )
            

        print("len of trigger times (Measure node):", len(trigger_times[0]), len(trigger_times[1]), "MEAS num_rejects:", num_rejects_0, num_rejects_1)

        return return_res


class RamanCharacterizationNode(Node):
    # This node was used to charecterize raman scattering. No EPS sources
    # were used and simply classical communication and a detector were used.  
    def __init__(self, name: str, timeline: "Timeline", params):
        super().__init__(name, timeline)

        self.detections = 0

        self.detector = Detector(name = self.name+".detector", timeline = timeline, efficiency = params["BSM_DET1_EFFICIENCY"], dark_count = params["DARK_COUNTS"], time_resolution=params["RESOLUTION"], temporal_coincidence_window = params["TEMPORAL_COINCIDENCE_WINDOW"], dead_time=params["DEAD_TIME"])
        self.add_component(self.detector)
        self.detector.attach(self)

        self.first_component_name = self.detector.name

        self.fiber_spool = add_classical_channel(self, self, timeline, distance = params["DISTANCE"], params = params)

    def start(self):
        self.fiber_spool.start_classical_communication()

    def trigger(self, detector, info):
        self.detections += 1


class WDMRelayNode(Node):
    # This is a relay node used in Jodan's experiment. It essentially takes a photon from the 
    # source and sends it down a fiber with classial coexistence in the fiber. 
    def __init__(self, name, timeline, bsm_name, start_classical_communication = False):
        super().__init__(name, timeline)
        self.bsm_name = bsm_name
        self.start_classical_communication = start_classical_communication

    def get(self, photon: "Photon", **kwargs):
        # print("WDMRelay node sending")
        if self.start_classical_communication and not self.cchannels[self.receiver].classical_communication_running:
            self.cchannels[self.bsm_name].start_classical_communication()
        self.send_qubit(self.bsm_name, photon)


class ValidationNode(Node):
    # (This may require debugging. The simulation was never validated.)
    # This simuates the node from Jordan's experiment. This node first 
    # creates a pair of photons and then transmimts one straight to a detector 
    # (BSM) and transmits the other through the fiber, to a measurement node (meas_node).  
    def __init__(self, name: str, timeline: "Timeline", params, BSM, meas_name):
        super().__init__(name, timeline)

        self.meas_name = meas_name
        self.BSM = BSM

        self.spdc_name = name + ".spdc_source"
        spdc = SPDCSource(self.spdc_name, timeline, wavelengths=[params["RETAINED_WAVELENGTH"], params["QUANTUM_WAVELENGTH"]],
                          frequency=params["SPDC_FREQUENCY"], mean_photon_num=params["MEAN_PHOTON_NUM"], polarization_fidelity=params['POLARIZATION_FIDELITY'])

        self.add_component(spdc)

        self.params = params

        self.num_output = params["MODE_NUM"]
        spdc.add_receiver(self)
        spdc.add_receiver(self)

    def start(self):
        states = [None] * self.num_output
        self.components[self.spdc_name].emit(states)

    def get(self, photon: "Photon", **kwargs):
        if photon.wavelength == self.params["RETAINED_WAVELENGTH"]:
            self.BSM.get(photon, src = self.name)
        else:
            self.send_qubit(self.meas_name, photon)


class PolarizationDistributionNode(Node):
    # This node is used to simulate Anirudh's experiment. This is a test 
    # of SeQUeNCe's legacy polarization encodding model and a test bed for 
    # testing new implementations of polarization. 
    def __init__(self, name: str, timeline: "Timeline", signal_node_name, idler_node_name, params):
        super().__init__(name, timeline)

        self.detections = 0

        self.num_output = params["MODE_NUM"]

        self.idler_node_name = idler_node_name
        self.signal_node_name = signal_node_name

        self.spdc_name = name + ".spdc_source"
        spdc = SPDCSource(self.spdc_name, timeline, wavelengths=[params["QUANTUM_WAVELENGTH"], params["QUANTUM_WAVELENGTH"]],
                          frequency=params["SPDC_FREQUENCY"], mean_photon_num=params["MEAN_PHOTON_NUM"], encoding_type=polarization, polarization_fidelity=params["POLARIZATION_FIDELITY"])

        self.add_component(spdc)
        spdc.add_receiver(self)
        spdc.add_receiver(self)
    def start(self):
        states = [[1/np.sqrt(2),1/np.sqrt(2)]] * self.num_output
        self.components[self.spdc_name].emit(states)

    def set_source_mpn(self, mpn):
        self.components[self.spdc_name].mean_photon_num = mpn

    def get(self, photon: "Photon", **kwargs):
        # print("sending photons at node.")
        if photon.name == "signal":
            # print("sending signal photon")
            self.send_qubit(self.signal_node_name, photon)
        elif photon.name == "idler":
            # print("sending idler photon")
            self.send_qubit(self.idler_node_name, photon)

class PolarizationReceiverNode(Node):
    def __init__(self, name, timeline, params):
        super().__init__(name, timeline)
        
        self.signal_detector = Detector(name = self.name+".signal_detector", timeline = timeline, efficiency = params["SIGNAL_DET_EFFICIENCY"], dark_count = params["SIGNAL_DET_DARK"], time_resolution=params["RESOLUTION"], temporal_coincidence_window = params["TEMPORAL_COINCIDENCE_WINDOW"], dead_time=params["SIGNAL_DET_DEAD"])
        self.add_component(self.signal_detector)
        self.signal_detector.attach(self)

        self.idler_detector = Detector(name = self.name+".idler_detector", timeline = timeline, efficiency = params["IDLER_DET_EFFICIENCY"], dark_count = params["IDLER_DET_DARK"], time_resolution=params["RESOLUTION"], temporal_coincidence_window = params["TEMPORAL_COINCIDENCE_WINDOW"], dead_time=params["IDLER_DET_DEAD"])
        self.add_component(self.idler_detector)
        self.idler_detector.attach(self)

        self.detections = {self.signal_detector:[], self.idler_detector:[]}
        self.temporal_coincidence_window = params["TEMPORAL_COINCIDENCE_WINDOW"]

        #  = Polarizer(name = self.name+".polarizer", timeline = timeline)
        self.two_qubit_polarizer = Polarizer("two_qubit_polarizer", timeline, num_qubits=2)
        self.one_qubit_polarizer = Polarizer("one_qubit_polarizer", timeline, num_qubits=1)

        self.add_component(self.two_qubit_polarizer)
        self.add_component(self.one_qubit_polarizer)

        self.add_component(self)
        self.first_component_name = self.name

        self.default_rng = np.random.default_rng()
        self.second_rng = np.random.default_rng()

        self.two_qubit_basis =  (
                                    (complex(1), complex(0), complex(0), complex(0)),
                                    (complex(0), complex(1), complex(0), complex(0)),
                                    (complex(0), complex(0), complex(1), complex(0)),
                                    (complex(0), complex(0), complex(0), complex(1))
                                )
        self.one_qubit_basis = (
                                    (complex(0), complex(1)),
                                    (complex(1), complex(0))
                               )   

        self.signal_polarizer_angle = 0
        self.idler_polarizer_angle = 0
        self.temp_photon = None
        self.is_idler = None
        self.temp_photon_time = None
        # self.singles_idler = 0
        # self.singles_signal = 0

        # self.det_idler_singles_count = 0
        # self.det_signal_singles_count = 0
        # self.coincidence_count = 0
        # self.rejects_signal = 0
        # self.rejects_idler = 0

        
    def set_det_eff(self, eff):
        self.signal_detector.efficiency = eff
        self.idler_detector.efficiency = eff



    def rotateSignal(self, angle):
        self.signal_polarizer_angle = angle
        self.detections[self.idler_detector].append([])
        self.detections[self.signal_detector].append([])

    def rotateIdler(self, angle):
        self.idler_polarizer_angle = angle

    def reset(self):
        self.signal_polarizer_angle = 0
        self.idler_polarizer_angle = 0
        self.temp_photon = None
        self.is_idler = None
        self.temp_photon_time = None

        self.det_idler_singles_count = 0
        self.det_signal_singles_count = 0
        self.coincidence_count = 0

        self.detections = {self.signal_detector:[], self.idler_detector:[]}


    def get(self, photon, is_idler = False):
        now = self.timeline.now()
        # print("getting qubit at", now)
        if (now < self.signal_detector.next_detection_time) and (now < self.idler_detector.next_detection_time):
            # print("both detectors blocked")
            return
        
        if self.temp_photon == None: # This is the first photon you are receiving
            self.temp_photon = photon
            self.original_state = self.temp_photon.quantum_state.state.copy()
            self.is_idler = is_idler
            self.temp_photon_time = now
        else:  
            # print("photons received at times:", now, self.temp_photon_time)          
            # print("second state before loss:")
            # print(self.temp_photon.quantum_state.state)
            photon.apply_loss()
            # print("incoming photons:")
            # print(photon.quantum_state)

            # print(self.temp_photon.quantum_state)

            # print("coincidence state")
            # print(self.temp_photon.quantum_state.state)
            # print(photon.quantum_state.state)
            # print("rotating 2 qubits:")          
            self.two_qubit_polarizer.rotate({0:self.signal_polarizer_angle, 1:self.idler_polarizer_angle})
            self.two_qubit_polarizer.get(photon)

            states, probabilities = FreeQuantumState.measure_multiple(self.two_qubit_basis, [self.temp_photon.quantum_state, photon.quantum_state], self.default_rng, return_states=True)
            # coincidence_prob = sum(probabilities)
            coincidence_prob = sum(probabilities)

            # print("coincidence_prob:", coincidence_prob)

            if self.default_rng.random() < coincidence_prob:
                # print("we have coincidence")
#################                self.coincidence_count += 1
                if self.default_rng.random() < self.signal_detector.efficiency:
                    self.signal_detector.record_detection()
                if self.default_rng.random() < self.idler_detector.efficiency:
                    self.idler_detector.record_detection()

            else:
                self.temp_photon.set_state(self.original_state, density_matrix = True)
                # print("state after setting") 
                # print(self.temp_photon.quantum_state.state)
                self.temp_photon.quantum_state.split_states()

                # print("state before sending to apply loss:")
                # print(self.temp_photon.quantum_state.state)
                # print(photon.quantum_state.state)

                photon.apply_loss()
                self.temp_photon.apply_loss()

                # print("quantum states:", photon.quantum_state, self.temp_photon.quantum_state)

                # print("state before sending to apply loss:")
                # print(self.temp_photon.quantum_state.state)
                # print(photon.quantum_state.state)
                # print("single polarization rotation:")
                if self.is_idler:
                    self.one_qubit_polarizer.rotate({0:self.signal_polarizer_angle})
                    self.one_qubit_polarizer.get(self.temp_photon)

                    self.one_qubit_polarizer.rotate({0:self.idler_polarizer_angle})
                    self.one_qubit_polarizer.get(photon)
                    det_idler_max_singles_prob = Photon.measure(self.one_qubit_basis, self.temp_photon, self.default_rng, return_prob = True)
                    det_signal_max_singles_prob = Photon.measure(self.one_qubit_basis, photon, self.default_rng, return_prob = True)
                else:
                    self.one_qubit_polarizer.rotate({0:self.idler_polarizer_angle})
                    self.one_qubit_polarizer.get(self.temp_photon)

                    self.one_qubit_polarizer.rotate({0:self.signal_polarizer_angle})
                    self.one_qubit_polarizer.get(photon)
                    det_idler_max_singles_prob = Photon.measure(self.one_qubit_basis, photon, self.default_rng, return_prob = True)
                    det_signal_max_singles_prob = Photon.measure(self.one_qubit_basis, self.temp_photon, self.default_rng, return_prob = True)

                # print("det_idler_max_singles_prob:", det_idler_max_singles_prob, "det_signal_max_singles_prob", det_signal_max_singles_prob)

                det_idler_remaining_prob = np.real((det_idler_max_singles_prob-coincidence_prob)/(1-coincidence_prob))
                det_signal_remaining_prob = np.real((det_signal_max_singles_prob-coincidence_prob)/(1-coincidence_prob))
                # print("det_idler_remaining_prob", det_idler_remaining_prob, "det_signal_remaining_prob", det_signal_remaining_prob)
                # print("det_signal_remaining_prob", det_signal_remaining_prob)
                # print("det_signal_max_singles_prob", det_signal_max_singles_prob)
                # print("coincidence_prob", coincidence_prob)
                # print("det_idler_remaining_prob", det_idler_remaining_prob, "det_signal_remaining_prob", det_signal_remaining_prob)
                # print()
                # single = self.default_rng.choice([1,2,0], p = [det_idler_remaining_prob, det_signal_remaining_prob, 1-det_signal_remaining_prob-det_idler_remaining_prob])
                # print("sample:", sample, "det_idler_remaining_prob", det_idler_remaining_prob, "det_signal_remaining_prob", det_signal_remaining_prob)

                # if self.default_rng.random() < 0.3:
                #     if now > self.idler_detector.next_detection_time:
                #         self.idler_detector.record_detection()
                #     else:
                #         self.signal_detector.record_detection()

                # print("det_idler_remaining_prob:", det_idler_remaining_prob, "det_signal_remaining_prob", det_signal_remaining_prob)

                # self.default_rng = np.random.default_rng()
                sample = self.default_rng.random()
                # print("Initial:", sample, "at time:", now)

                # print("self.idler_detector.temporal_coincidence_window", self.idler_detector.temporal_coincidence_window)
                # print("self.signal_detector.temporal_coincidence_window", self.signal_detector.temporal_coincidence_window)
                # print("self.idler_detector.dead_time", self.idler_detector.dead_time)
                # print("self.signal_detector.dead_time", self.signal_detector.dead_time)
                


                
                # print("idler_detector.next_detection_time", self.idler_detector.next_detection_time, "now:", now)
                # print("signal_detector.next_detection_time", self.signal_detector.next_detection_time, "now:", now)

                # if now < self.idler_detector.next_detection_time:
                #     self.rejects_idler += 1
                # self.idler_detector.record_detection()
                # self.singles_idler += 1

                # if now < self.signal_detector.next_detection_time:
                #     self.rejects_signal += 1
                # self.signal_detector.record_detection()
                # self.singles_signal += 1




                if sample > det_idler_remaining_prob and sample < (det_idler_remaining_prob+det_signal_remaining_prob):
                    # new_sample = 
                    # print("Signal:", new_sample)
                    if self.default_rng.random() < self.signal_detector.efficiency:
                        # print("detection success")
                        # if now < self.signal_detector.next_detection_time:
                        #     self.rejects_signal += 1
                        self.signal_detector.record_detection()
                        # self.singles_signal += 1

                elif sample < det_idler_remaining_prob:
                    # new_sample = 
                    # print("idler:", new_sample)
                    if self.default_rng.random() < self.idler_detector.efficiency:
                        # print("detection success")
                        # if now < self.idler_detector.next_detection_time:
                        #     self.rejects_idler += 1
                        self.idler_detector.record_detection()
                        # self.singles_idler += 1
                else:
                    print("problem!!!!!")

            self.temp_photon = None
            self.is_idler = None
            self.temp_photon_time = None
            # print()

    def trigger(self, detector, info):
        self.detections[detector][-1].append(info["time"])

    def get_data(self):
        coincidences = []
        coincidence_times = [] # only for debugging
        signal_detections = self.detections[self.signal_detector]
        idler_detections = self.detections[self.idler_detector]
        temporal_coincidence_window = self.temporal_coincidence_window
        # print("len:", len(signal_detections[0]), "signal_detections:", signal_detections)
        # print("len:", len(idler_detections[0]), "idler_detections", idler_detections)
        for signal, idler in zip(signal_detections, idler_detections):
            coincidence_count = 0
            idler_index = 0
            for i in signal:
                while idler_index < len(idler) and idler[idler_index] < i-temporal_coincidence_window:
                    idler_index += 1
                if idler_index >= len(idler):
                    break
                elif abs(i-idler[idler_index]) < temporal_coincidence_window:
                    coincidence_count += 1
                    coincidence_times.append(idler[idler_index])
                    coincidence_times.append(i)
            coincidences.append(coincidence_count)

        signal_singles = list(map(len, signal_detections))
        idler_singles = list(map(len, idler_detections))
        # print("coincidences:", coincidence_times)

        return signal_singles, idler_singles, coincidences


class proxyReceiver(Node):
    def __init__(self, name, timeline, receiver):
        super().__init__(name, timeline)
        self.add_component(self)
        self.receiver = receiver
        self.first_component_name = self.name

    def get(self, qubit):
        self.receiver.get(qubit, is_idler = True)