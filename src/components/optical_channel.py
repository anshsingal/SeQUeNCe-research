"""Models for simulation of optical fiber channels.

This module introduces the abstract OpticalChannel class for general optical fibers.
It also defines the QuantumChannel class for transmission of qubits/photons and the ClassicalChannel class for transmission of classical control messages.
OpticalChannels must be attached to nodes on both ends.
"""

import heapq as hq
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline
    from ..topology.node import Node
    from ..components.photon import Photon
    from ..message import Message

from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process


class OpticalChannel(Entity):
    """Parent class for optical fibers.

    Attributes:
        name (str): label for channel instance.
        timeline (Timeline): timeline for simulation.
        sender (Node): node at sending end of optical channel.
        receiver (Node): node at receiving end of optical channel.
        atteunuation (float): attenuation of the fiber (in dB/km).
        distance (int): length of the fiber (in m).
        polarization_fidelity (float): probability of no polarization error for a transmitted qubit.
        light_speed (float): speed of light within the fiber (in m/ps).
    """

    def __init__(self, name: str, timeline: "Timeline", attenuation: float, distance: int, polarization_fidelity: float, light_speed: float):
        """Constructor for abstract Optical Channel class.

        Args:
            name (str): name of the beamsplitter instance.
            timeline (Timeline): simulation timeline.
            attenuation (float): loss rate of optical fiber (in dB/km).
            distance (int): length of fiber (in m).
            polarization_fidelity (float): probability of no polarization error for a transmitted qubit.
            light_speed (float): speed of light within the fiber (in m/ps).
        """

        Entity.__init__(self, name, timeline)
        self.sender = None
        self.receiver = None
        self.attenuation = attenuation
        self.distance = distance  # (measured in m)
        self.polarization_fidelity = polarization_fidelity
        self.light_speed = light_speed # used for photon timing calculations (measured in m/ps)
        # self.chromatic_dispersion = kwargs.get("cd", 17)  # measured in ps / (nm * km)

    def init(self) -> None:
        pass

    def set_distance(self, distance: int) -> None:
        self.distance = distance


class QuantumChannel(OpticalChannel):
    """Optical channel for transmission of photons/qubits.

    Attributes:
        name (str): label for channel instance.
        timeline (Timeline): timeline for simulation.
        sender (Node): node at sending end of optical channel.
        receiver (Node): node at receiving end of optical channel.
        atteunuation (float): attenuation of the fiber (in dB/km).
        distance (int): length of the fiber (in m).
        polarization_fidelity (float): probability of no polarization error for a transmitted qubit.
        light_speed (float): speed of light within the fiber (in m/ps).
        loss (float): loss rate for transmitted photons (determined by attenuation).
        delay (int): delay (in ps) of photon transmission (determined by light speed, distance).
        max_rate (float): maximum rate of qubit transmission (in Hz).
    """

    def __init__(self, name: str, timeline: "Timeline", quantum_channel_attenuation: float, classical_channel_attenuation: float, distance: int, raman_coefficient, 
                 polarization_fidelity=1, light_speed=3e8, max_rate=1e12, quantum_channel_wavelength = 1536, classical_channel_wavelength = 1610):
        """Constructor for Quatnum Channel class.

        Args:
            name (str): name of the quantum channel instance.
            timeline (Timeline): simulation timeline.
            attenuation (float): loss rate of optical fiber (in dB/km).
            distance (int): length of fiber (in m).
            polarization_fidelity (float): probability of no polarization error for a transmitted qubit (default 1).
            light_speed (float): speed of light within the fiber (in m/ps) (default 2e-4).
            max_rate (float): maximum max_rate of qubit transmission (in Hz) (default 8e7).
        """

        super().__init__(name, timeline, quantum_channel_attenuation, distance, polarization_fidelity, light_speed)
        self.quantum_channel_attenuation = quantum_channel_attenuation
        self.classical_channel_attenuation = classical_channel_attenuation
        self.quantum_channel_wavelength = quantum_channel_wavelength
        self.classical_channel_wavelength = classical_channel_wavelength
        self.raman_coefficient = raman_coefficient
        self.delay = 0
        self.loss = 1
        self.max_rate = max_rate  # maximum rate for sending qubits (measured in Hz)
        self.send_bins = []

    def init(self) -> None:
        """Implementation of Entity interface (see base class)."""
        # print("optical channel is inited")
        self.delay = round((self.distance*1000 / self.light_speed) * 1e12)
        self.loss = 1 - 10 ** (self.distance * self.attenuation / -10)

    def set_ends(self, sender: "Node", receiver: str) -> None:
        """Method to set endpoints for the quantum channel.

        This must be performed before transmission.

        Args:
            sender (Node): node sending qubits.
            receiver (str): name of node receiving qubits.
        """
        self.sender = sender
        self.receiver = receiver
        sender.assign_qchannel(self, receiver)

    def start_clock(self, clock_power, narrow_band_filter_bandwidth):

        print("clock_started")

        # clock_power = 0.0003
        # raman_coefficient = 4.6 * 10**(-10)
        # narrow_band_filter_bandwidth = 1*10**(-6)
        # clock_attenuation = 0.099
        h = 6.62607015 * 10**(-28)
        c = 3 * 10**8

        calculation_window = 1/1000 # amount of time that one call of start clock caters to. Units in seconds. Not taking entire second since
                                    # no. of photons per second is extremey high. 

        raman_power = np.abs(clock_power * self.raman_coefficient * narrow_band_filter_bandwidth * (np.exp(-self.attenuation * self.distance) - np.exp(-self.classical_channel_attenuation * self.distance)) / (self.attenuation - self.classical_channel_attenuation))
        raman_power *= calculation_window
        num_photons_added = int(raman_power / (h * c / self.quantum_channel_wavelength))
        print(num_photons_added)
        raman_photon_distance_offset = np.floor((self.distance * np.random.rand(num_photons_added) / (3 * 10**5)) * 1e12)
        raman_photon_time_offset = np.floor(1e12 * calculation_window * np.random.rand(num_photons_added))
        raman_photon_arrival_times = self.timeline.now() + raman_photon_distance_offset + raman_photon_time_offset
        
        # for i in raman_photon_arrival_times[:1000]:
        #     print("Raman time offset:", i)
        # print("Raman photon time offset:", raman_photon_time_offset)
        for i in raman_photon_arrival_times:
            process = Process(self.receiver, "receive_qubit", [self.sender.name, None])
            event = Event(i, process)
            self.timeline.schedule(event)

        process = Process(self, "start_clock", [clock_power, narrow_band_filter_bandwidth])
        event = Event(self.timeline.now() + 1e12*calculation_window, process)
        self.timeline.schedule(event)
            

    def transmit(self, qubit, source: "Node") -> None:
        """Method to transmit photon-encoded qubits.

        Args:
            qubit (Photon): photon to be transmitted.
            source (Node): source node sending the qubit.

        Side Effects:
            Receiver node may receive the qubit (via the `receive_qubit` method).
        """

        assert self.delay != 0 and self.loss != 1, "QuantumChannel init() function has not been run for {}".format(self.name)
        assert source == self.sender

        # print("Photon being transmitted over fiber")

        # remove lowest time bin
        if len(self.send_bins) > 0:
            time = -1
            while time < self.timeline.now():
                time_bin = hq.heappop(self.send_bins)
                time = int(time_bin * (1e12 / self.max_rate))
            assert time == self.timeline.now(), "qc {} transmit method called at invalid time".format(self.name)

        # check if photon kept
        if (self.sender.get_generator().random() > self.loss) or qubit.is_null:
            # check if polarization encoding and apply necessary noise
            if (qubit.encoding_type["name"] == "polarization") and (
                    self.sender.get_generator().random() > self.polarization_fidelity):
                qubit.random_noise(self.get_generator())

            # schedule receiving node to receive photon at future time determined by light speed
            future_time = self.timeline.now() + self.delay
            process = Process(self.receiver, "receive_qubit",
                              [source.name, qubit])
            event = Event(future_time, process)
            self.timeline.schedule(event)

        # if photon lost, exit
        else:
            # print("photon lost in transmission")
            pass

    def schedule_transmit(self, min_time: int) -> int:
        """Method to schedule a time for photon transmission.

        Quantum Channels are limited by max_rate of transmission.
        This method returns the next available time for transmitting a photon.
        
        Args:
            min_time (int): minimum simulation time for transmission.

        Returns:
            int: simulation time for next available transmission window.
        """

        min_time = max(min_time, self.timeline.now())
        time_bin = min_time * (self.max_rate / 1e12)
        if time_bin - int(time_bin) > 0.00001:
            time_bin = int(time_bin) + 1
        else:
            time_bin = int(time_bin)

        # find earliest available time bin
        while time_bin in self.send_bins:
            time_bin += 1
        hq.heappush(self.send_bins, time_bin)

        # calculate time
        time = int(time_bin * (1e12 / self.max_rate))
        return time


class ClassicalChannel(OpticalChannel):
    """Optical channel for transmission of classical messages.

    Classical message transmission is assumed to be lossless.

    Attributes:
        name (str): label for channel instance.
        timeline (Timeline): timeline for simulation.
        sender (Node): node at sending end of optical channel.
        receiver (Node): node at receiving end of optical channel.
        distance (float): length of the fiber (in m).
        light_speed (float): speed of light within the fiber (in m/ps).
        delay (float): delay in message transmission (default distance / light_speed).
    """

    def __init__(self, name: str, timeline: "Timeline", distance: int, delay=-1):
        """Constructor for Classical Channel class.

        Args:
            name (str): name of the classical channel instance.
            timeline (Timeline): simulation timeline.
            distance (int): length of the fiber (in m).
            delay (float): delay (in ps) of message transmission (default distance / light_speed).
        """

        super().__init__(name, timeline, 0, distance, 0, 2e-4)
        if delay == -1:
            self.delay = distance / self.light_speed
        else:
            self.delay = delay

    def set_ends(self, sender: "Node", receiver: str) -> None:
        """Method to set endpoints for the classical channel.

        This must be performed before transmission.

        Args:
            sender (Node): node sending classical messages.
            receiver (str): name of node receiving classical messages.
        """
        self.sender = sender
        self.receiver = receiver
        sender.assign_cchannel(self, receiver)

    def transmit(self, message: "Message", source: "Node",
                 priority: int) -> None:
        """Method to transmit classical messages.

        Args:
            message (Message): message to be transmitted.
            source (Node): node sending the message.
            priority (int): priority of transmitted message (to resolve message reception conflicts).

        Side Effects:
            Receiver node may receive the qubit (via the `receive_qubit` method).
        """

        assert source == self.sender

        future_time = round(self.timeline.now() + int(self.delay))
        process = Process(self.receiver, "receive_message", [source.name, message])
        event = Event(future_time, process, priority)
        self.timeline.schedule(event)
