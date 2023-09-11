from typing import Callable
import numpy as np

from src.kernel.timeline import Timeline
from src.components.detector import QSDetectorFockDirect, QSDetectorFockInterference
from src.components.light_source import SPDCSource
from src.components.memory import AbsorptiveMemory
from src.components.photon import Photon
from src.topology.node import Node
from src.protocol import Protocol
from src.kernel.quantum_utils import *  # only for manual calculation and should not be used in simulation
from src.components.optical_channel import QuantumChannel, ClassicalChannel


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
                # print("classical dst is:", self.bsm_name)
                self.cchannels[self.bsm_name].start_classical_communication()
            self.send_qubit(self.bsm_name, photon)
        else:
            # from memory: send to destination (measurement) node
            if not self.cchannels[dst].classical_communication_running:
                # print("classical dst is:", dst)
                self.cchannels[dst].start_classical_communication()
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

        # The detector_name is a misleading title. it is actually the BSM name. We are getting detections from both detectors.
        trigger_times = self.components[detector_name].get_photon_times()


        print("len of trigger times (BSM node):", len(trigger_times[0]), len(trigger_times[1]))

        return_res = [0] * num_bins

        num_rejects_0 = 0
        num_rejects_1 = 0

        for time in trigger_times[0]:
            closest_bin = int(round((time - start_time) * frequency * 1e-12))
            expected_time = (float(closest_bin) * 1e12 / frequency) + start_time
            if abs(expected_time - time) < self.resolution and 0 <= closest_bin < num_bins:
                return_res[closest_bin] += 1
            else:
                num_rejects_0 += 1

        for time in trigger_times[1]:
            closest_bin = int(round((time - start_time) * frequency * 1e-12))
            expected_time = (float(closest_bin) * 1e12 / frequency) + start_time
            if abs(expected_time - time) < self.resolution and 0 <= closest_bin < num_bins:
                return_res[closest_bin] += 2
            else:
                num_rejects_1 += 1

        print("BSM num_rejects:", num_rejects_0, num_rejects_1)

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
        # print("start time is:", start_time, "bin width:", 1/(frequency*1e12))
        trigger_times = self.components[detector_name].get_photon_times()
        print("len of trigger times (Measure node):", len(trigger_times[0]), len(trigger_times[1]))
        return_res = [0] * num_bins

        # print("num of detector triggers:", len(trigger_times[0]), len(trigger_times[1]))

        num_rejects_0 = 0
        num_rejects_1 = 0


        for time in trigger_times[0]:
            closest_bin = int(round((time - start_time) * frequency * 1e-12))
            expected_time = (float(closest_bin) * 1e12 / frequency) + start_time
            # print("closest bin:", closest_bin)
            
            if abs(expected_time - time) < self.resolution and 0 <= closest_bin < num_bins:
                return_res[closest_bin] += 1
            else:
                num_rejects_0 += 1
                # print("actual_time:", time, "expected_time", expected_time, "error:", abs(expected_time - time), "closest_bin:", closest_bin )
                # print("too far gone, closest bin:", closest_bin)


        for time in trigger_times[1]:
            closest_bin = int(round((time - start_time) * frequency * 1e-12))
            expected_time = (float(closest_bin) * 1e12 / frequency) + start_time
            if abs(expected_time - time) < self.resolution and 0 <= closest_bin < num_bins:
                return_res[closest_bin] += 2
            else:
                num_rejects_1 += 1
                # print("actual_time:", time, "expected_time", expected_time, "error:", abs(expected_time - time), "closest_bin:", closest_bin )
            

        print("MEAS num_rejects:", num_rejects_0, num_rejects_1)

        return return_res
