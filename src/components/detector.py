"""Models for photon detection devices.

This module models a single photon detector (SPD) for measurement of individual photons.
It also defines a QSDetector class, which combines models of different hardware devices to measure photon states in different bases.
QSDetector is defined as an abstract template and as implementaions for polarization and time bin qubits.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict
import numpy as np

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline
    from ..components.photon import Photon
    from typing import List
from ..components.pulse_train import PulseTrain
from ..components.beam_splitter import BeamSplitter
from ..components.switch import Switch
from ..components.interferometer import Interferometer
from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process
from ..utils.encoding import time_bin
import time 
import cupy as cp
import h5py
import sys
np.set_printoptions(threshold = sys.maxsize)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)


class PulseDetector(Entity):
    """Pulse detector device."""

    def __init__(self, own, name: str, timeline: "Timeline", collection_probability=0.2, dark_count_rate=100, dead_time=1e3,
                 time_resolution=150):
        Entity.__init__(self, name+"_detector", timeline)  # Detector is part of the QSDetector, and does not have its own name
        self.own = own
        self.collection_probability = collection_probability
        self.dark_count_rate = dark_count_rate  # measured in 1/s
        self.dead_time = dead_time  # measured in Hz
        self.time_resolution = time_resolution  # measured in ps
        self.next_detection_time = -1
        self.detector_buffer = []
        self.log_file = h5py.File(f"{self.own.name}_buffer", "w")
        self.index = 0

    def init(self):
        """Implementation of Entity interface (see base class)."""
        pass



    def get(self, pulse_window) -> None:
        """Method to receive a pulse window for measurement.
        
        """

        # A binomial random variable is used to simulate the number of photons in the photon pulse train based on the collection probability of the 
        # detection setup. 
        for pulse_train in pulse_window.source_train:
            loss_matrix = np.random.binomial(pulse_train.photon_counts, 1-self.collection_probability)
            pulse_train.add_loss(loss_matrix)

        # Add dark counts and send the data to be stired on the disk
        dark_counts_pulse_train = self.add_dark_count(pulse_window.source_train[0].train_duration)
        pulse_window.noise_train.append(dark_counts_pulse_train)
        self.add_to_detector_buffer(pulse_window)


    def add_to_detector_buffer(self, pulse_window):
        """ This method saves the data received from the detector to the disk to be post process later"""
        now = self.timeline.now()
        temp_detector = np.array([])

        for pulse_train in pulse_window.source_train:
            temp_detector = np.append(temp_detector, now + pulse_train.time_offsets)

        for pulse_train in pulse_window.noise_train:
            temp_detector = np.append(temp_detector, now + pulse_train.time_offsets)

        for pulse_train in pulse_window.other_trains:
            temp_detector = np.append(temp_detector, now + pulse_train.time_offsets)

        self.log_file.create_dataset(f"{pulse_window.ID}", data = temp_detector)
        if self.own.name == "signal_receiver":
            print("pulse window ID", pulse_window.ID)



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


    def notify(self, info: Dict[str, Any]):
        """Custom notify function (calls `trigger` method)."""
        for observer in self._observers:
            observer.trigger(self, info)


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

    def __init__(self, name: str, timeline: "Timeline", efficiency=0.9, dark_count=0, count_rate=int(25e6),
                 time_resolution=150):
        Entity.__init__(self, name, timeline)  # Detector is part of the QSDetector, and does not have its own name
        self.efficiency = efficiency
        self.dark_count = dark_count  # measured in 1/s
        self.count_rate = count_rate  # measured in Hz
        self.time_resolution = time_resolution  # measured in ps
        self.next_detection_time = -1
        self.photon_counter = 0

    def init(self):
        """Implementation of Entity interface (see base class)."""
        self.add_dark_count()

    def get(self, dark_get=False) -> None:
        """Method to receive a photon for measurement.

        Args:
            dark_get (bool): Signifies if the call is the result of a false positive dark count event.
                If true, will ignore probability calculations (default false).

        Side Effects:
            May notify upper entities of a detection event.
        """

        self.photon_counter += 1
        now = self.timeline.now()
        time = round(now / self.time_resolution) * self.time_resolution

        if (
                self.get_generator().random() < self.efficiency or dark_get) and now > self.next_detection_time:
            self.notify({'time': time})
            self.next_detection_time = now + (
                        1e12 / self.count_rate)  # period in ps

    def add_dark_count(self) -> None:
        """Method to schedule false positive detection events.

        Events are scheduled as a Poisson process.

        Side Effects:
            May schedule future `get` method calls.
            May schedule future calls to self.
        """

        if self.dark_count > 0:
            time_to_next = int(self.get_generator().exponential(
                1 / self.dark_count) * 1e12)  # time to next dark count
            time = time_to_next + self.timeline.now()  # time of next dark count

            process1 = Process(self, "add_dark_count", [])  # schedule photon detection and dark count add in future
            process2 = Process(self, "get", [True])
            event1 = Event(time, process1)
            event2 = Event(time, process2)
            self.timeline.schedule(event1)
            self.timeline.schedule(event2)

    def notify(self, info: Dict[str, Any]):
        """Custom notify function (calls `trigger` method)."""
        for observer in self._observers:
            observer.trigger(self, info)


class QSDetector(Entity, ABC):
    """Abstract QSDetector parent class.

    Provides a template for objects measuring qubits in different encoding schemes.

    Attributes:
        name (str): label for QSDetector instance.
        timeline (Timeline): timeline for simulation.
        detectors (List[Detector]): list of attached detectors.
        trigger_times (List[List[int]]): tracks simulation time of detection events for each detector.
    """

    def __init__(self, name: str, timeline: "Timeline"):
        Entity.__init__(self, name, timeline)
        self.detectors = []
        self.components = []
        self.trigger_times = []

    def init(self):
        for component in self.components:
            component.attach(self)
            component.owner = self.owner

    def update_detector_params(self, detector_id: int, arg_name: str, value: Any) -> None:
        self.detectors[detector_id].__setattr__(arg_name, value)

    @abstractmethod
    def get(self, photon: "Photon") -> None:
        """Abstract method for receiving photons for measurement."""

        pass

    def trigger(self, detector: Detector, info: Dict[str, Any]) -> None:
        detector_index = self.detectors.index(detector)
        self.trigger_times[detector_index].append(info['time'])

    def get_photon_times(self):
        return self.trigger_times

    @abstractmethod
    def set_basis_list(self, basis_list: "List", start_time: int, frequency: int) -> None:
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
        self.detectors = [Detector(name + ".detector" + str(i), timeline) for i in range(2)]
        self.splitter = BeamSplitter(name + ".splitter", timeline)
        self.splitter.set_receiver(0, self.detectors[0])
        self.splitter.set_receiver(1, self.detectors[1])
        
        self.components = [self.splitter, self.detectors[0], self.detectors[1]]
        self.trigger_times = [[], []]

    def init(self) -> None:
        """Implementation of Entity interface (see base class)."""

        assert len(self.detectors) == 2
        super().init()

    def get(self, photon: "Photon") -> None:
        """Method to receive a photon for measurement.

        Forwards the photon to the internal polariaztion beamsplitter.

        Arguments:
            photon (Photon): photon to measure.

        Side Effects:
            Will call `get` method of attached beamsplitter.
        """

        self.splitter.get(photon)

    def get_photon_times(self):
        times, self.trigger_times = self.trigger_times, [[], []]
        return times

    def set_basis_list(self, basis_list: "List", start_time: int, frequency: int) -> None:
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
        self.switch.set_detector(self.detectors[0])
        self.interferometer = Interferometer(name + ".interferometer", timeline, time_bin["bin_separation"])
        self.interferometer.set_receiver(0, self.detectors[1])
        self.interferometer.set_receiver(1, self.detectors[2])
        self.switch.set_interferometer(self.interferometer)

        self.components = [self.switch, self.interferometer] + self.detectors
        self.trigger_times = [[], [], []]

    def init(self):
        """Implementation of Entity interface (see base class)."""

        assert len(self.detectors) == 3
        super().init()

    def get(self, photon):
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

    def set_basis_list(self, basis_list: "List", start_time: int, frequency: int) -> None:
        self.switch.set_basis_list(basis_list, start_time, frequency)

    def update_interferometer_params(self, arg_name: str, value: Any) -> None:
        self.interferometer.__setattr__(arg_name, value)
