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
from ..components.pulse_train import PulseTrain
from .photon import Photon
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
        self.earliest_available_time = 0
        self.record_raman_photons_start = None
        self.clock_running = False

    def init(self) -> None:
        """Implementation of Entity interface (see base class)."""
        # print("optical channel is inited")
        self.delay = round((self.distance*1000 / self.light_speed) * 1e12)
        # self.loss = 1 - 10 ** (self.distance * self.attenuation / -10)
        self.loss = self.attenuation * self.distance

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
        self.clock_running = True
        self.record_raman_photons_start = self.timeline.now()
        self.clock_power = clock_power
        self.narrow_band_filter_bandwidth = narrow_band_filter_bandwidth


    def add_raman_photons(self, train_duration):
        h = 6.62607015 * 10**(-34)
        c = 3 * 10**8

        raman_power = np.abs(self.clock_power * self.raman_coefficient * self.narrow_band_filter_bandwidth * (np.exp(-self.attenuation * self.distance) - np.exp(-self.classical_channel_attenuation * self.distance)) / (self.attenuation - self.classical_channel_attenuation))
        raman_energy = raman_power * train_duration/1e12
        num_photons_added = int(raman_energy / (h * c / self.quantum_channel_wavelength))

        # cur_max = train_duration
        # n = num_photons_added
        # photon_generation_times = np.zeros(n)                       #generate an array x of size n

        # for i in range(n,0,-1):
        #     cur_max = cur_max*np.random.rand()**(1/i)
        #     photon_generation_times[i-1] = cur_max

        print("Raman photosns added", num_photons_added)

        photon_generation_times = np.random.rand(num_photons_added) * train_duration

        return PulseTrain(photon_generation_times, train_duration, self.quantum_channel_wavelength)



        # raman_photon_distance_offset = np.floor((self.distance * np.random.rand(num_photons_added) / (3 * 10**5)) * 1e12)
        # raman_photon_time_offset = np.floor(window_size * np.random.rand(num_photons_added))
        
        

        # for i in raman_photon_arrival_times:
        #     state = np.around(np.random.rand())
        #     new_photon0 = Photon(None, wavelength=self.quantum_channel_wavelength, quantum_state=(complex(state), complex(1-state)))
        #     process = Process(self.receiver, "receive_qubit", [self.sender.name, new_photon0])
        #     event = Event(i, process)
        #     self.timeline.schedule(event)

        # process = Process(self, "start_clock", [clock_power, narrow_band_filter_bandwidth])
        # event = Event(self.timeline.now() + 1e12*calculation_window, process)
        # self.timeline.schedule(event)
            

    # def transmit(self, qubit, source: "Node") -> None:
    #     """Method to transmit photon-encoded qubits.

    #     Args:
    #         qubit (Photon): photon to be transmitted.
    #         source (Node): source node sending the qubit.

    #     Side Effects:
    #         Receiver node may receive the qubit (via the `receive_qubit` method).
    #     """

    #     assert self.delay != 0 and self.loss != 1, "QuantumChannel init() function has not been run for {}".format(self.name)
    #     assert source == self.sender

    #     # print("Photon being transmitted over fiber")

    #     # remove lowest time bin
    #     if len(self.send_bins) > 0:
    #         time = -1
    #         while time < self.timeline.now():
    #             time_bin = hq.heappop(self.send_bins)
    #             time = int(time_bin * (1e12 / self.max_rate))
    #         assert time == self.timeline.now(), "qc {} transmit method called at invalid time".format(self.name)

    #     # check if photon kept
    #     if (self.sender.get_generator().random() > self.loss) or qubit.is_null:
    #         # check if polarization encoding and apply necessary noise
    #         if (qubit.encoding_type["name"] == "polarization") and (
    #                 self.sender.get_generator().random() > self.polarization_fidelity):
    #             qubit.random_noise(self.get_generator())

    #         # schedule receiving node to receive photon at future time determined by light speed
    #         future_time = self.timeline.now() + self.delay
    #         process = Process(self.receiver, "receive_qubit",
    #                           [source.name, qubit])
    #         event = Event(future_time, process)
    #         self.timeline.schedule(event)

    #     # if photon lost, exit
    #     else:
    #         # print("photon lost in transmission")
    #         pass

    # def schedule_transmit(self, min_time: int) -> int:
    #     """Method to schedule a time for photon transmission.

    #     Quantum Channels are limited by max_rate of transmission.
    #     This method returns the next available time for transmitting a photon.
        
    #     Args:
    #         min_time (int): minimum simulation time for transmission.

    #     Returns:
    #         int: simulation time for next available transmission window.
    #     """

    #     min_time = max(min_time, self.timeline.now())
    #     time_bin = min_time * (self.max_rate / 1e12)
    #     if time_bin - int(time_bin) > 0.00001:
    #         time_bin = int(time_bin) + 1
    #     else:
    #         time_bin = int(time_bin)

    #     # find earliest available time bin
    #     while time_bin in self.send_bins:
    #         time_bin += 1
    #     hq.heappush(self.send_bins, time_bin)

    #     # calculate time
    #     time = int(time_bin * (1e12 / self.max_rate))
    #     return time





    def transmit(self, pulse_window, source: "Node") -> None:
        """Method to transmit photon-encoded qubits.

        Args:
            qubit (Photon): photon to be transmitted.
            source (Node): source node sending the qubit.

        Side Effects:
            Receiver node may receive the qubit (via the `receive_qubit` method).
        """

        assert self.delay != 0 and self.loss != 1, "QuantumChannel init() function has not been run for {}".format(self.name)
        assert source == self.sender

        # print("transmitting a photon on", self.name, "at", self.timeline.now(), "o time:", self.timeline.now() + self.delay)

        # check if photon kept
        # if (self.sender.get_generator().random() > self.loss):
        #     # check if polarization encoding and apply necessary noise
        #     if (qubit.encoding_type["name"] == "polarization") and (
        #             self.sender.get_generator().random() > self.polarization_fidelity):
        #         qubit.random_noise(self.get_generator())

            # schedule receiving node to receive photon at future time determined by light speed
            # future_time = self.timeline.now() + self.delay
            # process = Process(self.receiver, "receive_qubit",
            #                   [source.name, qubit])
            # event = Event(future_time, process)
            # self.timeline.schedule(event)

        # # if photon lost, exit
        # else:
        #     # print("photon lost in transmission")
        #     pass

        loss_matrix = np.random.binomial(pulse_window.source_train[0].photon_counts, self.loss)
        pulse_window.source_train[0].add_loss(loss_matrix)
        # print(self.name, "loss matrix:", loss_matrix)
        # print("after adding loss:", pulse_window.trains[0].time_offsets)
        # pulse_trains = [pulse_train]

        # if self.name == 'signal_channel':
        #     print("num transmitting photon train:", len(pulse_train.photon_counts))
            # for i,j in zip(pulse_train.photon_counts, pulse_train.time_offsets):
            #     print(i, j)


        # if self.name == 'signal_channel':
        #     for i in loss_matrix:
        #         print("photons loss:", i)
        #     for i,j in zip(pulse_train.photon_counts, pulse_train.time_offsets):
        #         print("photon kept:", i, "at", j)
        if self.clock_running:
            raman_photon_train = self.add_raman_photons(pulse_window.source_train[0].train_duration)
            # print("raman photon train type:", type(raman_photon_train.time_offsets))
            pulse_window.noise_train.append(raman_photon_train)
            # if self.name == 'signal_channel':
                # print("num Raman photon train sent:", len(raman_photon_train.photon_counts))
                # for i,j in zip(raman_photon_train.photon_counts, raman_photon_train.time_offsets):
                    # print(i, j)

        # print("transmitting a photon on", self.name, "at", self.timeline.now(), "on time:", pulse_window.trains[0].train_duration + self.timeline.now() + self.delay)
        # print("pulse_trains:", pulse_trains)
        future_time = self.timeline.now() + self.delay
        # print("receiver:", getattr(self.receiver, "receive_qubit"), type(self.receiver))
        process = Process(self.receiver, "receive_qubit", [source.name, pulse_window])
        event = Event(future_time, process)
        self.timeline.schedule(event)


    def schedule_transmit(self, pulse_train_length) -> int:
        """Method to schedule a time for photon transmission.

        Quantum Channels are limited by max_rate of transmission.
        This method returns the next available time for transmitting a photon.
        
        Args:
            min_time (int): minimum simulation time for transmission.

        Returns:
            int: simulation time for next available transmission window.
        """

        scheduled_time = max(self.timeline.now(), self.earliest_available_time)

        self.earliest_available_time = scheduled_time + pulse_train_length

        return scheduled_time

        # min_time = max(min_time, self.timeline.now())
        # time_bin = min_time * (self.max_rate / 1e12)
        # if time_bin - int(time_bin) > 0.00001:
        #     time_bin = int(time_bin) + 1
        # else:
        #     time_bin = int(time_bin)

        # # find earliest available time bin
        # while time_bin in self.send_bins:
        #     time_bin += 1
        # hq.heappush(self.send_bins, time_bin)

        # # calculate time
        # time = int(time_bin * (1e12 / self.max_rate))
        # return time





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
