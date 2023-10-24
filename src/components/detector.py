"""Models for photon detection devices.

This module models a single photon detector (SPD) for measurement of individual photons.
It also defines a QSDetector class, which combines models of different hardware devices to measure photon states in different bases.
QSDetector is defined as an abstract template and as implementations for polarization and time bin qubits.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List
from numpy import eye, kron, exp, sqrt
from scipy.linalg import fractional_matrix_power
from math import factorial
import numpy as np
# import cupy as cp
# from numba import jit
import sys
import numpy.ma as ma
np.set_printoptions(threshold = sys.maxsize)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline

from .photon import Photon
from .beam_splitter import BeamSplitter
from .switch import Switch
from .interferometer import Interferometer
from .circuit import Circuit
from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process
from ..utils.encoding import time_bin
from ..components.pulse_train import PulseTrain


class Detector(Entity):
    """Single photon detector device.

    This class models a single photon detector, for detecting photons.
    Can be attached to many different devices to enable different measurement options.

    Attributes:
        name (str): label for detector instance.
        timeline (Timeline): timeline for simulation.
        efficiency (float): probability to successfully measure an incoming photon.
        dark_count (float): average number of false positive detections per second.
        count_rate (float): maximum detection rate; defines detector cooldown time.
        time_resolution (int): minimum resolving power of photon arrival time (in ps).
        photon_counter (int): counts number of detection events.
    """

    _meas_circuit = Circuit(1)
    _meas_circuit.measure(0)

    def __init__(self, name: str, timeline: "Timeline", efficiency=0.9, dark_count=0, count_rate=int(25e6),
                 time_resolution=10, temporal_coincidence_window = 450, dead_time = 7000):
        Entity.__init__(self, name, timeline)  # Detector is part of the QSDetector, and does not have its own name
        self.efficiency = efficiency
        self.dark_count = dark_count  # measured in 1/s
        self.count_rate = count_rate  # measured in Hz
        self.time_resolution = time_resolution  # measured in ps
        self.next_detection_time = -1
        self.photon_counter = 0
        self.temporal_coincidence_window = temporal_coincidence_window
        self.dead_time = dead_time

    def init(self):
        """Implementation of Entity interface (see base class)."""
        self.next_detection_time = -1
        self.photon_counter = 0
        if self.dark_count > 0:
            self.add_dark_count()

    def get(self, photon=None, **kwargs) -> None:
        """Method to receive a photon for measurement.

        Args:
            photon (Photon): photon to detect (currently unused)

        Side Effects:
            May notify upper entities of a detection event.
        """

        self.photon_counter += 1

        num_photons = 0
        if type(photon) == np.ndarray:
            # print("time of getting raman photons:", self.timeline.now())
            # print("photon is:")
            # print(photon)
            for pulse in photon:
                for detection_time in pulse:
                    if detection_time == 0:
                        break
                    num_photons += 1
                    process = Process(self, "record_detection", [])
                    event = Event(self.timeline.now()+detection_time, process)
                    self.timeline.schedule(event)
                    # print("Raman photon time:", self.timeline.now()+detection_time)
            print("number of raman photons are:", num_photons)
            return

        # if get a photon and it has single_atom encoding, measure
        if photon and photon.encoding_type["name"] == "single_atom":
            key = photon.quantum_state
            res = self.timeline.quantum_manager.run_circuit(Detector._meas_circuit, [key],
                                                            self.get_generator().random())
            # if we measure |0>, return (do not record detection)
            if not res[key]:
                return

        if self.get_generator().random() < self.efficiency:
            self.record_detection()

    def add_dark_count(self) -> None:
        """Method to schedule false positive detection events.

        Events are scheduled as a Poisson process.

        Side Effects:
            May schedule future `get` method calls.
            May schedule future calls to self.
        """

        assert self.dark_count > 0, "Detector().add_dark_count called with 0 dark count rate"
        time_to_next = int(self.get_generator().exponential(
                1 / self.dark_count) * 1e12)  # time to next dark count
        time = time_to_next + self.timeline.now()  # time of next dark count

        process1 = Process(self, "add_dark_count", [])  # schedule photon detection and dark count add in future
        process2 = Process(self, "record_detection", [])
        event1 = Event(time, process1)
        event2 = Event(time, process2)
        self.timeline.schedule(event1)
        self.timeline.schedule(event2)

    def record_detection(self):
        """Method to record a detection event.

        Will calculate if detection succeeds (by checking if we have passed `next_detection_time`)
        and will notify observers with the detection time (rounded to the nearest multiple of detection frequency).
        """

        now = self.timeline.now()
        # print("recording measurement at detector class")
        # print("recording detection at:", now)

        if now > self.next_detection_time:
            time = round(now / self.time_resolution) * self.time_resolution
            self.notify({'time': time})
            self.next_detection_time = now + self.dead_time # (1e12 / self.count_rate)  # Multiplied by 4 to give us a dead time of 5ns since SPDC frequency is 8e8 -> 1250ps dead time. 
            # print("detection at:", now, "next detection time:", self.next_detection_time)
        # else:
        #     print("rejected in dead time")
            # if self.name == "Eckhardt Research Center Measurement3.bs.detector1":
                # print("Dark count reject at detector:", self.name)

    def notify(self, info: Dict[str, Any]):
        """Custom notify function (calls `trigger` method)."""
        # self._observers list is populated using the attach method of the entity class. 
        for observer in self._observers:
            observer.trigger(self, info)


class QSDetector(Entity, ABC):
    """Abstract QSDetector parent class.

    Provides a template for objects measuring qubits in different encoding schemes.

    Attributes:
        name (str): label for QSDetector instance.
        timeline (Timeline): timeline for simulation.
        components (List[entity]): list of all aggregated hardware components.
        detectors (List[Detector]): list of attached detectors.
        trigger_times (List[List[int]]): tracks simulation time of detection events for each detector.
    """

    def __init__(self, name: str, timeline: "Timeline"):
        Entity.__init__(self, name, timeline)
        self.components = []
        self.detectors = []
        self.trigger_times = []

    def init(self):
        # The init method is run when we call timeline.init() for all the entities in the simulator. 
        for component in self.components:
            component.attach(self)
            component.owner = self.owner

    def update_detector_params(self, detector_id: int, arg_name: str, value: Any) -> None:
        self.detectors[detector_id].__setattr__(arg_name, value)

    @abstractmethod
    def get(self, photon: "Photon", **kwargs) -> None:
        """Abstract method for receiving photons for measurement."""

        pass

    def trigger(self, detector: Detector, info: Dict[str, Any]) -> None:
        # TODO: rewrite
        # print("detector was triggered")
        if self.name == "Eckhardt Research Center Measurement3.bs.detector1":
            print("detection at time", info['time'])
        detector_index = self.detectors.index(detector)
        self.trigger_times[detector_index].append(info['time'])

    def set_detector(self, idx: int,  efficiency=0.9, dark_count=0, count_rate=int(25e6), time_resolution=150):
        """Method to set the properties of an attached detector.

        Args:
            idx (int): the index of attached detector whose properties are going to be set.
            For other parameters see the `Detector` class. Default values are same as in `Detector` class.
        """
        assert 0 <= idx < len(self.detectors), "`idx` must be a valid index of attached detector."

        detector = self.detectors[idx]
        detector.efficiency = efficiency
        detector.dark_count = dark_count
        detector.count_rate = count_rate
        detector.time_resolution = time_resolution

    def get_photon_times(self):
        return self.trigger_times

    @abstractmethod
    def set_basis_list(self, basis_list: List[int], start_time: int, frequency: float) -> None:
        pass


class QSDetectorPolarization(QSDetector):
    """QSDetector to measure polarization encoded qubits.

    There are two detectors.
    Detectors[0] and detectors[1] are directly connected to the beamsplitter.

    Attributes:
        name (str): label for QSDetector instance.
        timeline (Timeline): timeline for simulation.
        detectors (List[Detector]): list of attached detectors (length 2).
        trigger_times (List[List[int]]): tracks simulation time of detection events for each detector.
        splitter (BeamSplitter): internal beamsplitter object.
    """

    def __init__(self, name: str, timeline: "Timeline"):
        QSDetector.__init__(self, name, timeline)
        for i in range(2):
            d = Detector(name + ".detector" + str(i), timeline)
            self.detectors.append(d)
            d.attach(self)
        self.splitter = BeamSplitter(name + ".splitter", timeline)
        self.splitter.add_receiver(self.detectors[0])
        self.splitter.add_receiver(self.detectors[1])
        self.trigger_times = [[], []]

        self.components = [self.splitter] + self.detectors

    def init(self) -> None:
        """Implementation of Entity interface (see base class)."""

        assert len(self.detectors) == 2
        super().init()

    def get(self, photon: "Photon", **kwargs) -> None:
        """Method to receive a photon for measurement.

        Forwards the photon to the internal polariaztion beamsplitter.

        Arguments:
            photon (Photon): photon to measure.

        Side Effects:
            Will call `get` method of attached beamsplitter.
        """

        self.splitter.get(photon)

    def get_photon_times(self):
        times = self.trigger_times
        self.trigger_times = [[], []]
        return times

    def set_basis_list(self, basis_list: List[int], start_time: int, frequency: float) -> None:
        self.splitter.set_basis_list(basis_list, start_time, frequency)

    def update_splitter_params(self, arg_name: str, value: Any) -> None:
        self.splitter.__setattr__(arg_name, value)


class QSDetectorTimeBin(QSDetector):
    """QSDetector to measure time bin encoded qubits.

    There are three detectors.
    The switch is connected to detectors[0] and the interferometer.
    The interferometer is connected to detectors[1] and detectors[2].

    Attributes:
        name (str): label for QSDetector instance.
        timeline (Timeline): timeline for simulation.
        detectors (List[Detector]): list of attached detectors (length 3).
        trigger_times (List[List[int]]): tracks simulation time of detection events for each detector.
        switch (Switch): internal optical switch component.
        interferometer (Interferometer): internal interferometer component.
    """

    def __init__(self, name: str, timeline: "Timeline"):
        QSDetector.__init__(self, name, timeline)
        self.switch = Switch(name + ".switch", timeline)
        self.detectors = [Detector(name + ".detector" + str(i), timeline) for i in range(3)]
        self.switch.add_receiver(self.detectors[0])
        self.interferometer = Interferometer(name + ".interferometer", timeline, time_bin["bin_separation"])
        self.interferometer.add_receiver(self.detectors[1])
        self.interferometer.add_receiver(self.detectors[2])
        self.switch.add_receiver(self.interferometer)

        self.components = [self.switch, self.interferometer] + self.detectors
        self.trigger_times = [[], [], []]

    def init(self):
        """Implementation of Entity interface (see base class)."""

        assert len(self.detectors) == 3
        super().init()

    def get(self, photon, **kwargs):
        """Method to receive a photon for measurement.

        Forwards the photon to the internal fiber switch.

        Args:
            photon (Photon): photon to measure.

        Side Effects:
            Will call `get` method of attached switch.
        """

        self.switch.get(photon)

    def get_photon_times(self):
        times, self.trigger_times = self.trigger_times, [[], [], []]
        return times

    def set_basis_list(self, basis_list: List[int], start_time: int, frequency: float) -> None:
        self.switch.set_basis_list(basis_list, start_time, frequency)

    def update_interferometer_params(self, arg_name: str, value: Any) -> None:
        self.interferometer.__setattr__(arg_name, value)


class QSDetectorFockDirect(QSDetector):
    """QSDetector to directly measure photons in Fock state.

    Usage: to measure diagonal elements of effective density matrix.

    Attributes:
        name (str): label for QSDetector instance.
        timeline (Timeline): timeline for simulation.
        src_list (List[str]): list of two sources which send photons to this detector (length 2).
        detectors (List[Detector]): list of attached detectors (length 2).
        trigger_times (List[List[int]]): tracks simulation time of detection events for each detector.
        arrival_times (List[List[int]]): tracks simulation time of Photon arrival at each input port
    """

    def __init__(self, name: str, timeline: "Timeline", src_list: List[str]):
        super().__init__(name, timeline)
        assert len(src_list) == 2
        self.src_list = src_list

        for i in range(2):
            d = Detector(name + ".detector" + str(i), timeline)
            self.detectors.append(d)
        self.components = self.detectors

        self.trigger_times = [[], []]
        self.arrival_times = [[], []]

        self.povms = [None] * 4

    def init(self):
        self._generate_povms()
        super().init()

    def _generate_povms(self):
        """Method to generate POVM operators corresponding to photon detector having 0 and 1 click
        Will be used to generated outcome probability distribution.
        """

        # assume using Fock quantum manager
        truncation = self.timeline.quantum_manager.truncation
        create, destroy = self.timeline.quantum_manager.build_ladder()

        create0 = create * sqrt(self.detectors[0].efficiency)
        destroy0 = destroy * sqrt(self.detectors[0].efficiency)
        series_elem_list = [((-1)**i) * fractional_matrix_power(create0, i+1).dot(
            fractional_matrix_power(destroy0, i+1)) / factorial(i+1) for i in range(truncation)]
        povm0_1 = sum(series_elem_list)
        povm0_0 = eye(truncation+1) - povm0_1

        create1 = create * sqrt(self.detectors[1].efficiency)
        destroy1 = destroy * sqrt(self.detectors[1].efficiency)
        series_elem_list = [((-1) ** i) * fractional_matrix_power(create1, i + 1).dot(
            fractional_matrix_power(destroy1, i + 1)) / factorial(i + 1) for i in range(truncation)]
        povm1_1 = sum(series_elem_list)
        povm1_0 = eye(truncation + 1) - povm0_1

        self.povms = [povm0_0, povm0_1, povm1_0, povm1_1]

    def get(self, photon: "Photon", **kwargs):

        if type(photon) == np.ndarray:
            for pulse in photon:
                for detection_time in pulse:
                    if detection_time == 0:
                        break
                    process = Process(self.detectors[round(np.random.rand())], "record_detection", [])
                    event = Event(self.timeline.now()+detection_time, process)
                    self.timeline.schedule(event)
            return

        src = kwargs["src"]
        assert photon.encoding_type["name"] == "fock", "Photon must be in Fock representation."
        input_port = self.src_list.index(src)  # determine at which input the Photon arrives, an index

        # record arrival time
        arrival_time = self.timeline.now()
        self.arrival_times[input_port].append(arrival_time)

        key = photon.quantum_state  # the photon's key pointing to the quantum state in quantum manager
        samp = self.get_generator().random()  # random measurement sample
        if input_port == 0:
            result = self.timeline.quantum_manager.measure([key], self.povms[0:2], samp)
        elif input_port == 1:
            result = self.timeline.quantum_manager.measure([key], self.povms[2:4], samp)
        else:
            raise Exception("too many input ports for QSDFockDirect {}".format(self.name))

        assert result in list(range(len(self.povms))), "The measurement outcome is not valid."
        if result == 1:
            # trigger time recording will be done by SPD
            self.detectors[input_port].record_detection()

    def get_photon_times(self) -> List[List[int]]:
        trigger_times = self.trigger_times
        self.trigger_times = [[], []]
        return trigger_times

    # does nothing for this class
    def set_basis_list(self, basis_list: List[int], start_time: int, frequency: int) -> None:
        pass


class QSDetectorFockInterference(QSDetector):
    """QSDetector with two input ports and two photon detectors behind beamsplitter.

    The detectors will physically measure the two beamsplitter output  photonic modes' Fock states, respectively.
    POVM operators which apply to pre-beamsplitter photonic state are used.
    NOTE: in the current implementation, to realize interference, we require that Photons arrive at both input ports
    simultaneously, and at most 1 Photon instance can be input at an input port at a time.

    Usage: to realize Bell state measurement (BSM) and to measure off-diagonal elements of the effective density matrix.

    Attributes:
        name (str): label for QSDetector instance.
        timeline (Timeline): timeline for simulation.
        src_list (List[str]): list of two sources which send photons to this detector (length 2).
        detectors (List[Detector]): list of attached detectors (length 2).
        phase (float): relative phase between two input optical paths.
        trigger_times (List[List[int]]): tracks simulation time of detection events for each detector.
        detect_info (List[List[Dict]]): tracks detection information, including simulation time of detection events
            and detection outcome for each detector.
        arrival_times (List[List[int]]): tracks simulation time of arrival of photons at each input mode.

        temporary_photon_info (List[Dict]): temporary list of information of Photon arriving at each input port.
            Specific to current implementation. At most 1 Photon's information will be recorded in a dictionary.
            When there are 2 non-empty dictionaries,
            e.g. [{"photon":Photon1, "time":arrival_time1}, {"photon":Photon2, "time":arrival_time2}],
            the entangling measurement will be carried out. After measurement, the temporary list will be reset.
    """

    def __init__(self, name: str, timeline: "Timeline", src_list: List[str], phase: float = 0):
        super().__init__(name, timeline)
        assert len(src_list) == 2
        self.src_list = src_list
        self.phase = phase

        for i in range(2):
            d = Detector(name + ".detector" + str(i), timeline)
            self.detectors.append(d)
        self.components = self.detectors

        self.trigger_times = [[], []]
        self.detect_info = [[], []]
        self.arrival_times = [[], []]
        self.temporary_photon_info = [{}, {}]

        self.povms = [None] * 4

    def init(self):
        self._generate_povms()
        super().init()

    def _generate_transformed_ladders(self):
        """Method to generate transformed creation/annihilation operators by the beamsplitter.

        Will be used to construct POVM operators.
        """

        truncation = self.timeline.quantum_manager.truncation
        identity = eye(truncation + 1)
        create, destroy = self.timeline.quantum_manager.build_ladder()
        phase = self.phase
        efficiency1 = sqrt(self.detectors[0].efficiency)
        efficiency2 = sqrt(self.detectors[1].efficiency)

        # Modified mode operators in Heisenberg picture by beamsplitter transformation
        # considering inefficiency and ignoring relative phase
        create1 = (kron(efficiency1*create, identity) + exp(1j*phase)*kron(identity, efficiency2*create)) / sqrt(2)
        destroy1 = create1.conj().T
        create2 = (kron(efficiency1*create, identity) - exp(1j*phase)*kron(identity, efficiency2*create)) / sqrt(2)
        destroy2 = create2.conj().T

        return create1, destroy1, create2, destroy2

    # def _generate_povms(self):
    #     """Method to generate POVM operators corresponding to photon detector having 00, 01, 10 and 11 click(s).

    #     Will be used to generated outcome probability distribution.
    #     """

    #     # assume using Fock quantum manager
    #     truncation = self.timeline.quantum_manager.truncation
    #     create1, destroy1, create2, destroy2 = self._generate_transformed_ladders()

    #     # print("dimensions of create1:", type(create1), len(create1))

    #     # for detector1 (index 0)
        
    #     # In effect, this is: 
    #     #   -1^(i) * [ a_dagger^(i+1) @ a^(i+1) ] / (i+1)! 
    #     # and, we sum this over all possible number of photons n in the truncation assumed. See paper. Formula used directly.  
    #     series_elem_list1 = [(-1)**i * fractional_matrix_power(create1, i+1).dot(
    #         fractional_matrix_power(destroy1, i+1)) / factorial(i+1) for i in range(truncation)]
        
    #     povm1_1 = sum(series_elem_list1)
    #     povm0_1 = eye((truncation+1) ** 2) - povm1_1


        
    #     # for detector2 (index 1)
    #     series_elem_list2 = [(-1)**i * fractional_matrix_power(create2, i+1).dot(
    #         fractional_matrix_power(destroy2,i+1)) / factorial(i+1) for i in range(truncation)]
    #     povm1_2 = sum(series_elem_list2)
    #     povm0_2 = eye((truncation+1) ** 2) - povm1_2

    #     # POVM operators for 4 possible outcomes
    #     # Note: povm01 and povm10 are relevant to BSM
    #     povm00 = povm0_1 @ povm0_2
    #     povm01 = povm0_1 @ povm1_2
    #     povm10 = povm1_1 @ povm0_2
    #     povm11 = povm1_1 @ povm1_2

    #     # print("dimensions of POVMs:", type(povm11), len(povm11))

    #     self.povms = [povm00, povm01, povm10, povm11]


    def _generate_povms(self):
        """Method to generate POVM operators corresponding to photon detector having 00, 01, 10 and 11 click(s).

        Will be used to generated outcome probability distribution.
        """

        # assume using Fock quantum manager
        truncation = self.timeline.quantum_manager.truncation
        create1, destroy1, create2, destroy2 = self._generate_transformed_ladders()

        # print("dimensions of create1:", type(create1), len(create1))

        # for detector1 (index 0)
        
        # In effect, this is: 
        #   -1^(i) * [ a_dagger^(i+1) @ a^(i+1) ] / (i+1)! 
        # and, we sum this over all possible number of photons n in the truncation assumed. See paper. Formula used directly.  
        series_elem_list1 = [(-1)**i * fractional_matrix_power(create1, i+1).dot(
            fractional_matrix_power(destroy1, i+1)) / factorial(i+1) for i in range(truncation)]
        
        povm1_1 = sum(series_elem_list1)
        povm0_1 = eye((truncation+1) ** 2) - povm1_1


        
        # for detector2 (index 1)
        series_elem_list2 = [(-1)**i * fractional_matrix_power(create2, i+1).dot(
            fractional_matrix_power(destroy2,i+1)) / factorial(i+1) for i in range(truncation)]
        povm1_2 = sum(series_elem_list2)
        povm0_2 = eye((truncation+1) ** 2) - povm1_2

        identity = eye((truncation+1) ** 2)
        

        # POVM operators for 4 possible outcomes (When both detectors active)
        # Note: povm01 and povm10 are relevant to BSM
        povm00 = povm0_1 @ povm0_2
        povm01 = povm0_1 @ povm1_2
        povm10 = povm1_1 @ povm0_2
        povm11 = povm1_1 @ povm1_2

        # POVM when detector 1 off:
        povm_0 = identity @ povm0_2
        povm_1 = identity @ povm1_2

        # POVM when detector 2 off:
        povm0_ = povm0_1 @ identity
        povm1_ = povm1_1 @ identity

        # POVM when both detectors are off:
        povm = identity

        self.povms = [[povm00, povm01, povm10, povm11], [povm0_, povm1_], [povm_0, povm_1], [povm]]



    def get(self, photon, **kwargs):
        # print("type of array:", type(photon))
        num_photons = 0
        if type(photon) == np.ndarray:
            # print("time of getting raman photons:", self.timeline.now())
            # print("photon is:")
            # print(photon)
            for pulse in photon:
                for detection_time in pulse:
                    if detection_time == 0:
                        break
                    num_photons += 1
                    process = Process(self.detectors[round(np.random.rand())], "record_detection", [])
                    event = Event(self.timeline.now()+detection_time, process)
                    self.timeline.schedule(event)
                    # print("Raman photon time:", self.timeline.now()+detection_time)
            print("number of raman photons are:", num_photons)
            return
        # print("photon arrived for detection")
            # detector_number = np.random.choice([0,1], len(pulse.photon_counts))
            # masked_photon_counts = ma.masked_array(pulse.photon_counts, mask = detector_number)
            # masked_time_offsets = ma.masked_array(pulse.time_offsets, mask = detector_number)
            
            # detector_0_pulse_train = PulseTrain()
            # detector_0_pulse_train.time_offsets = masked_time_offsets[masked_time_offsets.mask]
            # detector_0_pulse_train.photon_counts = masked_photon_counts[masked_photon_counts.mask]
            
            # detector_1_pulse_train = PulseTrain()
            # detector_1_pulse_train.time_offsets = masked_time_offsets[~masked_time_offsets.mask]
            # detector_1_pulse_train.photon_counts = masked_photon_counts[~masked_photon_counts.mask]
            
            # self.detectors[0].schedule_arrivals(detector_0_pulse_train)
            # self.detectors[1].schedule_arrivals(detector_1_pulse_train)


            # print("its a numpy aray")
        # print("new photon arrived")
        # print("regula photon time:", self.timeline.now())
        src = kwargs["src"]
        # print("photon received from", src)
        assert photon.encoding_type["name"] == "fock", "Photon must be in Fock representation."
        input_port = self.src_list.index(src)  # determine at which input the Photon arrives, an index
        # record arrival time
        arrival_time = self.timeline.now()
        self.arrival_times[input_port].append(arrival_time)
        # record in temporary photon list
        assert not self.temporary_photon_info[input_port], \
            "At most 1 Photon instance should arrive at an input port at a time."
        self.temporary_photon_info[input_port]["photon"] = photon
        self.temporary_photon_info[input_port]["time"] = arrival_time

        # judge if there have already been two input Photons arriving simultaneously
        dict0 = self.temporary_photon_info[0]
        dict1 = self.temporary_photon_info[1]
        # if both two dictionaries are non-empty
        if dict0 and dict1:
            assert dict0["time"] == dict1["time"], "To realize interference photons must arrive simultaneously."
            photon0 = dict0["photon"]
            photon1 = dict1["photon"]
            key0 = photon0.quantum_state
            key1 = photon1.quantum_state

            # determine the outcome
            samp = self.get_generator().random()  # random measurement sample
            # print("dimensions of POVMs:", type(self.povms), len(self.povms[0]))
            # print("Arrival time:", arrival_time)
            index = 0
            if self.detectors[1].next_detection_time > arrival_time:
                index += 1
            if self.detectors[0].next_detection_time > arrival_time:
                index += 2    
            # print("index of detection:", index)
            # if index != 0:
                # print("index is:", index)

            verbose = False
            # if  self.name == "BSMNode.bsm": verbose = True

            result = self.timeline.quantum_manager.measure([key0, key1], self.povms[index], samp, verbose)

            assert result in list(range(len(self.povms))), "The measurement outcome is not valid."
            # print("self.name:", self.name)
            detection_time = self.timeline.now()
            if result == 0:
                # no click for either detector, but still record the zero outcome
                # record detection information
                
                info = {"time": detection_time, "outcome": 0}
                self.detect_info[0].append(info)
                self.detect_info[1].append(info)

            elif result == 1:
                # detector 1 has a click
                # trigger time recording will be done by SPD
                # if self.name == "Eckhardt Research Center Measurement3.bs":
                #     print("Reg detection 1 at", self.timeline.now())
                self.detectors[1].record_detection()
                # record detection information
                # detection_time = self.timeline.now()
                info0 = {"time": detection_time, "outcome": 0}
                info1 = {"time": detection_time, "outcome": 1}
                self.detect_info[0].append(info0)
                self.detect_info[1].append(info1)

            elif result == 2:
                # detector 0 has a click
                # trigger time recording will be done by SPD
                # if self.name == "Eckhardt Research Center Measurement3.bs":
                #     print("Reg detection 2 at", self.timeline.now())
                self.detectors[0].record_detection()
                # record detection information
                # detection_time = self.timeline.now()
                info0 = {"time": detection_time, "outcome": 1}
                info1 = {"time": detection_time, "outcome": 0}
                self.detect_info[0].append(info0)
                self.detect_info[1].append(info1)

            elif result == 3:
                # both detectors have a click
                # trigger time recording will be done by SPD
                # if self.name == "Eckhardt Research Center Measurement3.bs":
                #     print("Reg detection 1&2 at", self.timeline.now())
                self.detectors[0].record_detection()
                self.detectors[1].record_detection()
                # record detection information
                # detection_time = self.timeline.now()
                info = {"time": detection_time, "outcome": 1}
                self.detect_info[0].append(info)
                self.detect_info[1].append(info)

            self.temporary_photon_info = [{}, {}]

        else:
            pass

        """
        # check if we have non-null photon
        if not photon.is_null:
            state = self.timeline.quantum_manager.get(photon.quantum_state)

            # if entangled, apply phase gate
            if len(state.keys) == 2:
                self.timeline.quantum_manager.run_circuit(self._circuit, state.keys)

            self.beamsplitter.get(photon)
        """

    def get_photon_times(self) -> List[List[int]]:
        """Method to get detector trigger times and detection information.
        Will clear `trigger_times` and `detect_info`.
        """
        trigger_times = self.trigger_times
        # detect_info = self.detect_info
        self.trigger_times = [[], []]
        self.detect_info = [[], []]
        # return trigger_times, detect_info
        # if self.name == "Eckhardt Research Center Measurement3.bs":
        # print("trigger times where:", trigger_times)
        return trigger_times

    # does nothing for this class
    def set_basis_list(self, basis_list: List[int], start_time: int, frequency: float) -> None:
        pass

    def set_phase(self, phase: float):
        self.phase = phase
        self._generate_povms()



class PULSE_Detector(Entity):
    """Pulse detector device."""

    def __init__(self, own, name: str, timeline: "Timeline", collection_probability=0.2, dark_count_rate=100, dead_time=0,
                 time_resolution=150):
        Entity.__init__(self, name+"_detector", timeline)  # Detector is part of the QSDetector, and does not have its own name
        self.own = own
        self.collection_probability = collection_probability
        self.dark_count_rate = dark_count_rate  # measured in 1/s
        self.dead_time = dead_time  # measured in Hz
        self.time_resolution = time_resolution  # measured in ps
        self.next_detection_time = -1
        self.detector_buffer = []
        # self.log_file = h5py.File(f"{self.own.name}_buffer", "w")
        self.index = 0
        self.prev_dead_time = 0
        print("detector owner name:", self.own.name)
        with open(f"{self.name}_buffer.dat", "w") as fileID:
            pass

    def init(self):
        """Implementation of Entity interface (see base class)."""
        pass


    def schedule_arrivals(self, pulse_train : PulseTrain):

        print("Scheduling arrivals of Raman photons")

        self.noise_pulse_train = pulse_train
        loss_matrix = np.random.binomial(self.noise_pulse_train.photon_counts, 1-self.collection_probability)
        self.noise_pulse_train.add_loss(loss_matrix)
        process = Process(self, "get", [None, True])
        for arrival_time in self.noise_pulse_train.time_offsets:
            event = Event(self.timeline.now()+arrival_time, process)
            self.timeline.schedule(event)


    def get(self, pulse, noise = False) -> None:
        """Method to receive a pulse window for measurement.
        
        """
        if noise:
            print("noisy photon was detected.")
        else:
            print("Qubit photon detected")
        if self.timeline.now() > self.next_detection_time:
            print("we are within detection window")
            if not noise and np.random.binomial(pulse.quantum_state, self.collection_probability) < 0:
                return
            print("processing detection at", self.timeline.now())
            self.next_detection_time = self.timeline.now() + self.dead_time
            self.notify({"time": self.timeline.now()})
            
        # for pulse_train in pulse_window.source_train:
        #     loss_matrix = np.random.binomial(pulse_train.photon_counts, 1-self.collection_probability)
        #     pulse_train.add_loss(loss_matrix)

        # # Add dark counts and send the data to be stored on the disk
        # dark_counts_pulse_train = self.add_dark_count(pulse_window.source_train[0].train_duration)
        # pulse_window.noise_train.append(dark_counts_pulse_train)
        # self.add_to_detector_buffer(pulse_window)


    # def add_to_detector_buffer(self, pulse_window):
    #     """ This method saves the data received from the detector to the disk to be post process later"""
    #     now = self.timeline.now()
    #     temp_detector = np.array([])

    #     for pulse_train in pulse_window.source_train:
    #         temp_detector = np.append(temp_detector, now + pulse_train.time_offsets)

    #     for pulse_train in pulse_window.noise_train:
    #         temp_detector = np.append(temp_detector, now + pulse_train.time_offsets)

    #     for pulse_train in pulse_window.other_trains:
    #         temp_detector = np.append(temp_detector, now + pulse_train.time_offsets)
            
    #     # print("sorting window", i)  
    #     # print("pulse train length: ", len(temp_detector))  
    #     temp_detector, self.prev_dead_time = self.sort_remove_dead_counts(temp_detector, self.dead_time, self.prev_dead_time)
    #     # idler_buffer, prev_idler_dead_time = self.sort_remove_dead_counts(self.idler_buffer[str(i)][:], self.idler_dead_time, prev_idler_dead_time)
    #     # print("sorted window", i)
    #     # print("dead time removal done")

    #     # print("Last detection:", temp_detector[-1])
    #     # print(type(temp_detector[-1]))

    #     if self.own.name == "idler_receiver":
    #         # print("pulse window ID", pulse_window.ID)
    #         print("type of arrival times:", type(temp_detector[0]))
    #         print("idler arrivals:", len(temp_detector))
    #         print(temp_detector)

    #     # self.log_file.create_dataset(f"{pulse_window.ID}", data = temp_detector)
    #     with open(f"{self.own.name}_buffer.dat", "ab+") as fileID:
    #         fileID.write(temp_detector.data)
    #     if self.own.name == "signal_receiver":
    #         print("pulse window ID", pulse_window.ID)



    def add_dark_count(self, duration) -> None:
        """Method to schedule false positive detection events.

        Events are scheduled as a Poisson process.

        """
        net_rate = (self.dark_count_rate/1e12) * duration
        num_photon_pairs = np.random.poisson(lam = net_rate)
        if num_photon_pairs == 0:
            return PulseTrain(np.array([]), duration, None)
        last_arrival = duration + 1
        while last_arrival > duration:
            last_arrival = np.random.gamma(shape = num_photon_pairs, scale = 1e12/self.dark_count_rate)

        arrival_times = np.random.rand(num_photon_pairs - 1) * duration
        arrival_times = np.append(arrival_times, [int(last_arrival)])
        return PulseTrain(arrival_times, duration, None)


    def sort_remove_dead_counts(self, pulse_train, dead_time, prev_dead_time):
        """ This method sorts the detections and removes the detections which are too close for the detector dead time"""
        
        # Commenting out to remove warnings about Jit and cupy

        # def GPU_sort(pulse_train):
        #     GPU_pulse_train = cp.asarray(pulse_train)
        #     GPU_sorted_pulse_train = cp.sort(GPU_pulse_train)
        #     return cp.asnumpy(GPU_sorted_pulse_train)

        # Sorting the array using a GPU for performance. Intsead of sorting this, we can merge this using parallelised algorithms
        # sorted_pulse_train = GPU_sort(pulse_train)

        # Remove the detections which lie in the dead time of the previous batch
        i = 0
        for i in range(len(pulse_train)):
            if pulse_train[i] > prev_dead_time:
                break
        sorted_pulse_train = sorted_pulse_train[i:]

        # Removal of dark counts is done by JIT compiling the actual method and executing the kernel.
        # @jit(parallel = True, nopython = True)
        def remove_dark_counts(sorted_pulse_train):
            mask = np.ones(len(sorted_pulse_train))
            i = 0
            while i<=len(sorted_pulse_train)-1:
                mask[i] = 0
                j = 1
                while len(sorted_pulse_train) > i+j and sorted_pulse_train[i+j] <= sorted_pulse_train[i] + dead_time:
                    j = j+1
                i = i + j
            return mask

        mask = remove_dark_counts(sorted_pulse_train)
        
        sorted_pulse_train = ma.masked_array(sorted_pulse_train, mask = mask)
        out = sorted_pulse_train[~sorted_pulse_train.mask]
        # print("done with dark count removal")
        return out, out[-1] + dead_time




    def notify(self, info: Dict[str, Any]):
        print("notification at", info, "at time:", self.timeline.now())
        """Custom notify function (calls `trigger` method)."""
        # for observer in self._observers:
        #     observer.trigger(self, info)
