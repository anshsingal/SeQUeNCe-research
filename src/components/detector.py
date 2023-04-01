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

# class PulseDetector(Entity):
#     """Pulse detector device."""

#     def __init__(self, own, name: str, timeline: "Timeline", collection_probability=0.2, dark_count_rate=100, dead_time=1e3,
#                  time_resolution=150):
#         Entity.__init__(self, name+"_detector", timeline)  # Detector is part of the QSDetector, and does not have its own name
#         self.own = own
#         self.collection_probability = collection_probability
#         self.dark_count_rate = dark_count_rate  # measured in 1/s
#         self.dead_time = dead_time  # measured in Hz
#         self.time_resolution = time_resolution  # measured in ps
#         self.next_detection_time = -1
#         # self.photon_counter = 0

#     def init(self):
#         """Implementation of Entity interface (see base class)."""
#         self.add_dark_count()

#     def get(self, pulse = None) -> None:
#         # print("photon received by detector")
#         """Method to receive a photon for measurement.

#         Args:
#             dark_get (bool): Signifies if the call is the result of a false positive dark count event.
#                 If true, will ignore probability calculations (default false).

#         Side Effects:
#             May notify upper entities of a detection event.
#         """
#         # self.photon_counter += 1
#         now = self.timeline.now()

#         # if pulse:
#         #     print(f"detector get time: {now}")
#         # print("detection recieved, pulse:", type(pulse), "at time", now)
#         time = round(now / self.time_resolution) * self.time_resolution
#         if now > self.next_detection_time:
#             if pulse == None: # Dark count
#                 # time = round(now / self.time_resolution) * self.time_resolution
#                 self.notify({'time': time})
#             else:
#                 time = now
#                 # for i in pulse:
#                 if np.random.rand() < self.collection_probability:
#                     # time = int((now)/self.time_resolution) * self.time_resolution
#                     self.notify({'time': time})
            

#             self.next_detection_time = now + self.dead_time  # period in ps

#     def add_dark_count(self) -> None:
#         """Method to schedule false positive detection events.

#         Events are scheduled as a Poisson process.

#         Side Effects:
#             May schedule future `get` method calls.
#             May schedule future calls to self.
#         """

#         if self.dark_count_rate > 0:
#             time_to_next = int(self.get_generator().exponential(
#                 1 / self.dark_count_rate) * 1e12)  # time to next dark count
#             time = time_to_next + self.timeline.now()  # time of next dark count

#             process1 = Process(self, "add_dark_count", [])  # schedule photon detection and dark count add in future
#             process2 = Process(self, "get", [True])
#             event1 = Event(time, process1)
#             event2 = Event(time, process2)
#             self.timeline.schedule(event1)
#             self.timeline.schedule(event2)

#     def notify(self, info: Dict[str, Any]):
#         """Custom notify function (calls `trigger` method)."""
#         for observer in self._observers:
#             observer.trigger(self, info)


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
        # self.f = open(f"{self.own.name}_buffer.txt", "w+")
        self.log_file = h5py.File(f"{self.own.name}_buffer", "w")
        self.index = 0
        # self.photon_counter = 0

    def init(self):
        """Implementation of Entity interface (see base class)."""
        # self.add_dark_count()
        pass

    # def add_to_buffer(self, pulse_trains):


    def get(self, pulse_window) -> None:
        # print("photon received by detector")
        """Method to receive a photon for measurement.

        Args:
            dark_get (bool): Signifies if the call is the result of a false positive dark count event.
                If true, will ignore probability calculations (default false).

        Side Effects:
            May notify upper entities of a detection event.
        """
        # self.photon_counter += 1
        
        # print("detected a photon train at", self.own.name, now)

        # if pulse:
        #     print(f"detector get time: {now}")
        # print("detection recieved, pulse:", type(pulse), "at time", now)
        # time = round(now / self.time_resolution) * self.time_resolution

        for pulse_train in pulse_window.trains:
            # print("Type of photon counts:", type(pulse_train.photon_counts[0]))
            loss_matrix = np.random.binomial(pulse_train.photon_counts, 1-self.collection_probability)
            pulse_train.add_loss(loss_matrix)

            # print(self.own.name, "final detected loss matrix:", loss_matrix)
            # print(self.own.name, "final detected:", pulse_train.time_offsets)

            

            # if self.own.name == "signal_receiver":
            #     for i,j in zip(pulse_train.photon_counts, self.detector_buffer[0]):
            #         print("photons detected:", i, "at", j)


        dark_counts_pulse_train = self.add_dark_count(pulse_window.trains[0].train_duration)
        # print("dark count type:", type(dark_counts_pulse_train.time_offsets))
        pulse_window.trains.append(dark_counts_pulse_train)

        # if self.own.name == "idler_receiver":
        #     print("num detected photon train:", len(pulse_trains[0].photon_counts))
        #     # for i,j in zip(pulse_trains[0].photon_counts, pulse_trains[0].time_offsets):
        #     #     print(i, j)
        #     print("num detected raman train:", len(pulse_trains[1].photon_counts))
        #     # for i,j in zip(pulse_trains[1].photon_counts, pulse_trains[1].time_offsets):
        #     #     print(i, j)
        #     print("num dark count train:", pulse_trains[2].photon_counts)
        #     # for i,j in zip(pulse_trains[2].photon_counts, pulse_trains[2].time_offsets):
        #     #     print(i, j)

        self.add_to_detector_buffer(pulse_window)

        

        # print(self.own.name, "raman potons:", pulse_trains[1].time_offsets)

        # CPU_start = time.time()
        # CPU_sorted_raman_array = sorted(pulse_trains[1].time_offsets)
        # CPU_end = time.time()

        # print("CPU sorting time:", CPU_end - CPU_start)

        # GPU_start = time.time()
        # GPU_raman_array = cp.asarray(pulse_trains[1].time_offsets)
        # GPU_sorted_raman_array = cp.sort(GPU_raman_array)
        # GPU_end = time.time()

        # print("GPU sorting time:", GPU_end - GPU_start)


        # print(pulse_trains)

        # self.add_to_buffer(time, pulse_trains)


    def add_to_detector_buffer(self, pulse_window):
        now = self.timeline.now()
        temp_detector = np.array([])

        for pulse_train in pulse_window.trains:
            # print(self.own.name, type(pulse_train.time_offsets))
            temp_detector = np.append(temp_detector, now + pulse_train.time_offsets)

        # print("length of pulse train detected in pulse ID:", pulse_window.ID, len(temp_detector))
        # print("pulse window ID", pulse_window.ID)
        # print(self.own.name, "window ID:", pulse_window.ID, temp_detector)
        self.log_file.create_dataset(f"{pulse_window.ID}", data = temp_detector)
        # print(self.own.name)
        if self.own.name == "signal_receiver":
            print("pulse window ID", pulse_window.ID)


        # self.notify({'time':})
            # print(self.own.name, "time offsets:", pulse_train.time_offsets, "with size:", len(pulse_train.time_offsets))
            # temp_detector = np.append(temp_detector, now + cp.asarray(pulse_train.time_offsets))
        
        # GPU_detection_array = cp.asarray(temp_detector)
        # GPU_sorted_detection_array = cp.sort(cp.asarray(temp_detector))
        # cp.cuda.runtime.deviceSynchronize()

        # list_str = str(cp.asnumpy(GPU_sorted_detection_array))
        # print("sorted list:", list_str)
        # self.f.write(list_str)
        # self.f.write("\n")
        # self.f.flush()
        # self.f.write(str(temp_detector))
        # self.f.write("\n")
        # self.f.flush()



        # self.detector_buffer.extend(cp.asnumpy(GPU_sorted_detection_array))






    def add_dark_count(self, duration) -> None:
        """Method to schedule false positive detection events.

        Events are scheduled as a Poisson process.

        Side Effects:
            May schedule future `get` method calls.
            May schedule future calls to self.
        """
        net_rate = (self.dark_count_rate/1e12) * duration
        num_photon_pairs = np.random.poisson(lam = net_rate)
        # num_photon_pairs = np.random.poisson(lam = (self.dark_count_rate/1e12) * duration)
        if num_photon_pairs == 0:
            return PulseTrain(np.array([]), duration, None)
            # return []
        last_arrival = duration + 1
        while last_arrival > duration:
            last_arrival = np.random.gamma(shape = num_photon_pairs, scale = 1e12/self.dark_count_rate)

        # print("dark count dura")
        # cur_max = last_arrival
        # n = num_photon_pairs-1
        # arrival_times = np.array([0]*(n))                  

        # for i in range(n,0,-1):
        #     cur_max = cur_max*np.random.rand()**(1/i)
        #     print("cu_max:", cur_max)
        #     arrival_times[i-1] = int(cur_max)
        arrival_times = np.random.rand(num_photon_pairs - 1) * duration
        
        arrival_times = np.append(arrival_times, [int(last_arrival)])

        # print("type of arrival_times:", type(arrival_times), arrival_times)

        return PulseTrain(arrival_times, duration, None)

        # if self.dark_count_rate > 0:
        #     time_to_next = int(self.get_generator().exponential(
        #         1 / self.dark_count_rate) * 1e12)  # time to next dark count
        #     time = time_to_next + self.timeline.now()  # time of next dark count

        #     process1 = Process(self, "add_dark_count", [])  # schedule photon detection and dark count add in future
        #     process2 = Process(self, "get", [True])
        #     event1 = Event(time, process1)
        #     event2 = Event(time, process2)
        #     self.timeline.schedule(event1)
        #     self.timeline.schedule(event2)

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
