"""Models for simulation of optical fiber channels.

This module introduces the abstract OpticalChannel class for general optical fibers.
It also defines the QuantumChannel class for transmission of qubits/photons and the ClassicalChannel class for transmission of classical control messages.
OpticalChannels must be attached to nodes on both ends.
"""

import heapq as hq
from typing import TYPE_CHECKING
import numpy as np
import re
from scapy.all import PcapNgReader, raw
from bitstring import BitArray

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline
    from ..topology.node import Node
    from ..message import Message

from ..components.photon import Photon
from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process
from ..utils import log
from ..components.pulse_train import PulseTrain


class OpticalChannel(Entity):
    """Parent class for optical fibers.

    Attributes:
        name (str): label for channel instance.
        timeline (Timeline): timeline for simulation.
        sender (Node): node at sending end of optical channel.
        receiver (Node): node at receiving end of optical channel.
        attenuation (float): attenuation of the fiber (in dB/m).
        distance (int): length of the fiber (in m).
        polarization_fidelity (float): probability of no polarization error for a transmitted qubit.
        light_speed (float): speed of light within the fiber (in m/ps).
    """

    def __init__(self, name: str, timeline: "Timeline", attenuation: float, distance: int,
                 polarization_fidelity: float, light_speed: float):
        """Constructor for abstract Optical Channel class.

        Args:
            name (str): name of the beamsplitter instance.
            timeline (Timeline): simulation timeline.
            attenuation (float): loss rate of optical fiber (in dB/m).
            distance (int): length of fiber (in m).
            polarization_fidelity (float): probability of no polarization error for a transmitted qubit.
            light_speed (float): speed of light within the fiber (in m/ps).
        """
        log.logger.info("Create channel {}".format(name))

        Entity.__init__(self, name, timeline)
        self.sender = None
        self.receiver = None
        self.attenuation = attenuation
        self.distance = distance  # (measured in m)
        self.polarization_fidelity = polarization_fidelity
        self.light_speed = light_speed  # used for photon timing calculations (measured in m/ps)
        # self.chromatic_dispersion = kwargs.get("cd", 17)  # measured in ps / (nm * km)

    def init(self) -> None:
        pass

    def set_distance(self, distance: int) -> None:
        self.distance = distance


class PULSE_QuantumChannel(OpticalChannel):
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
                 polarization_fidelity=1, light_speed=3e8, max_rate=1e12, quantum_channel_wavelength = 1536, classical_channel_wavelength = 1610, window_size = 1e11, frequency=8e7):
        """Constructor for Quatnum Channel class.

        Args:
            name (str): name of the quantum channel instance.
            timeline (Timeline): simulation timeline.
            quantum_channel_attenuation (float): loss rate in the quantum band of the optical fiber.
            classical_channel_attenuation (float): loss rate in the classical band of the optical fiber.
            distance (int): length of fiber (in km).
            raman_coefficient (float): Raman coefficient of the fiber
            polarization_fidelity (float): probability of no polarization error for a transmitted qubit (default 1).
            light_speed (float): speed of light within the fiber (in m/ps) (default 2e-4).
            max_rate (float): maximum max_rate of qubit transmission (in Hz) (default 8e7).
            quantum_channel_wavelength (int): wavelength of the quantum communication band.
            classical_channel_wavelength (int): wavelength of the classical communication band.
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
        # self.clock_running = False
        self.frequency = frequency 
        # self.window_size = window_size
        

    def init(self) -> None:
        """Implementation of Entity interface (see base class)."""
        self.delay = round((self.distance*1000 / self.light_speed) * 1e12)
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

    # def start_clock(self, clock_power, narrow_band_filter_bandwidth):
    #     """ Starts the add_raman_photons process """
    #     self.clock_running = True
    #     self.record_raman_photons_start = self.timeline.now()
    #     self.clock_power = clock_power
    #     self.narrow_band_filter_bandwidth = narrow_band_filter_bandwidth
    #     self.scheduled_raman_train = 0
    #     self.add_raman_train()


    def transmit_classical_message(self, pulse_times, pulse_width, direction, clock_power, narrow_band_filter_bandwidth):
        """ adds a photon train of noisy photons scattered from the classical band into the quantum band."""

        # Required parameters: Pulse width, to calculate raman power inside the pulse travelling through the fiber. 
        # This needs to be sent by the user. 

        h = 6.62607015 * 10**(-34)
        c = 3 * 10**8
        window_size = (self.distance * 1000 / c) # We are taking the distance to be in KM and hence, convert it to m first. Then,
                                                 # the time of exposure is simply the time required for light to cross the detector.
                                                 # Half pulses (during emission and detection) is ignored. 

        # print("pulse times are:", pulse_times)
                           
        if direction:
            print("Photon for True direction faced")
        else:
            print("Photon for False direction faced")
        
        pulse_width = (pulse_width/1e12) * c / 1e3 # We get the pulse width in picoseconds. Hence, we convert it 
                                                   # seconds and then convert it in the space domain using the speed of 
                                                   # light and convert that distance to km. 

        raman_power = np.abs(clock_power * self.raman_coefficient * narrow_band_filter_bandwidth * (np.exp(-self.attenuation * pulse_width) - np.exp(-self.classical_channel_attenuation * pulse_width)) / (self.attenuation - self.classical_channel_attenuation))
        
        print("clock power:", clock_power, "narrow_band_filter_bandwidth", narrow_band_filter_bandwidth, "exponent:", np.exp(-self.attenuation * pulse_width) - np.exp(-self.classical_channel_attenuation * pulse_width))
        
        raman_energy = raman_power * window_size
        mean_num_photons = (raman_energy / (h * c / self.quantum_channel_wavelength))
        print("mean_num_photons", mean_num_photons)
        dAlpha = self.attenuation - self.classical_channel_attenuation

        detection_times = []

        for pulse_time in pulse_times:
            num_photons_added = np.random.poisson(mean_num_photons)
            if num_photons_added > 0:
                generated_locations = np.random.uniform(0, self.distance, num_photons_added)
                probabilities_of_transmission = np.exp(-self.attenuation*self.distance)*(np.exp(dAlpha*generated_locations)-1) / (np.exp(-self.classical_channel_attenuation*self.distance) - np.exp(-self.attenuation*self.distance))
                decision_array = np.random.binomial(1, probabilities_of_transmission, len(probabilities_of_transmission))
                # Need to find some reference for the spectrum of light in fiber optics to get the classical and quantum channel speeds of in the fiber. For now, using the same c for both. 
                new_detections = np.array([(pulse_time + (location*1000 / c + (self.distance-location)*1000 / c) * 1e12) for decision, location in zip(decision_array, generated_locations) if decision])
                detection_times.extend(new_detections)

        print("detection times are:", detection_times)

        raman_photon_train = PulseTrain(detection_times, self.quantum_channel_wavelength)
        
        print("scheduling receive qubit at quantum channel: receiver:", self.receiver)
        process = Process(self.receiver, "receive_qubit", [self.sender.name, raman_photon_train])
        event = Event(self.timeline.now(), process)
        self.timeline.schedule(event)


        
        
        # Send detection times to the detector

    # def add_raman_train(self):
    #     if self.clock_running:
    #         print("scheduling Raman photon detection at", self.name)
    #         raman_photon_train = self.add_raman_photons()
            
    #         process = Process(self.receiver, "receive_qubit", [self.sender.name, raman_photon_train])
    #         event = Event(self.timeline.now(), process)
    #         self.timeline.schedule(event)

    #         self.scheduled_raman_train += self.window_size

    #         process = Process(self, "add_raman_train", [])
    #         event = Event(self.scheduled_raman_train, process)
    #         self.timeline.schedule(event)


            
        
            
            # pulse_window.noise_train.append(raman_photon_train)


    def transmit(self, photon: Photon, source: "Node") -> None:
        """Method to transmit photon-encoded qubits.

        Args:
            qubit (Photon): photon to be transmitted.
            source (Node): source node sending the qubit.

        Side Effects:
            Receiver node may receive the qubit (via the `receive_qubit` method).
        """

        assert self.delay != 0 and self.loss != 1, "QuantumChannel init() function has not been run for {}".format(self.name)
        # assert source == self.sender

        if len(self.send_bins) > 0:
            time = -1
            while time < self.timeline.now():
                time_bin = hq.heappop(self.send_bins)
                time = int(time_bin * (1e12 / self.frequency))
            assert time == self.timeline.now(), "qc {} transmit method called at invalid time".format(self.name)



        transmitted_photon_number = np.random.binomial(photon.quantum_state, 1-self.loss)
        # pulse_window.source_train[0].add_loss(loss_matrix)
        photon.quantum_state = transmitted_photon_number
        print(f"in optical channel {self.name}, sending photon with photon number", transmitted_photon_number)
       
        
        future_time = self.timeline.now() + self.delay

        process = Process(self.receiver, "receive_qubit", [source.name, photon])
        event = Event(future_time, process)
        self.timeline.schedule(event)

        process = Process(self.receiver, "internal_notification", [future_time])
        event = Event(self.timeline.now(), process)
        self.timeline.schedule(event)


    def schedule_transmit(self, min_time) -> int:
        """Method to schedule a time for photon transmission.

        Quantum Channels are limited by a frequency of transmission.
        This method returns the next available time for transmitting a photon.
        
        Args:
            min_time (int): minimum simulation time for transmission.

        Returns:
            int: simulation time for next available transmission window.
        """

        # TODO: move this to node?

        min_time = self.timeline.now()
        time_bin = min_time * (self.frequency / 1e12)
        if time_bin - int(time_bin) > 0.00001:
            time_bin = int(time_bin) + 1
        else:
            time_bin = int(time_bin)

        # find earliest available time bin
        while time_bin in self.send_bins:
            time_bin += 1
        hq.heappush(self.send_bins, time_bin)

        # calculate time
        time = int(time_bin * (1e12 / self.frequency))
        return time



class QuantumChannel(OpticalChannel):
    """Optical channel for transmission of photons/qubits.

    Attributes:
        name (str): label for channel instance.
        timeline (Timeline): timeline for simulation.
        sender (Node): node at sending end of optical channel.
        receiver (Node): node at receiving end of optical channel.
        attenuation (float): attenuation of the fiber (in dB/m).
        distance (int): length of the fiber (in m).
        polarization_fidelity (float): probability of no polarization error for a transmitted qubit.
        light_speed (float): speed of light within the fiber (in m/ps).
        loss (float): loss rate for transmitted photons (determined by attenuation).
        delay (int): delay (in ps) of photon transmission (determined by light speed, distance).
        frequency (float): maximum frequency of qubit transmission (in Hz).
    """

    def __init__(self, name: str, timeline: "Timeline", attenuation: float, distance: int,
                 polarization_fidelity=1.0, light_speed=2e-4, frequency=8e7):
        """Constructor for Quantum Channel class.

        Args:
            name (str): name of the quantum channel instance.
            timeline (Timeline): simulation timeline.
            attenuation (float): loss rate of optical fiber (in dB/m).
            distance (int): length of fiber (in m).
            polarization_fidelity (float): probability of no polarization error for a transmitted qubit (default 1).
            light_speed (float): speed of light within the fiber (in m/ps) (default 2e-4).
            frequency (float): maximum frequency of qubit transmission (in Hz) (default 8e7).
        """

        super().__init__(name, timeline, attenuation, distance, polarization_fidelity, light_speed)
        self.delay = -1
        self.loss = 1
        self.frequency = frequency  # maximum frequency for sending qubits (measured in Hz)
        self.send_bins = []

    def init(self) -> None:
        """Implementation of Entity interface (see base class)."""

        self.delay = round(self.distance / self.light_speed)
        self.loss = 1 - 10 ** (self.distance * self.attenuation / -10)

    def set_ends(self, sender: "Node", receiver: str) -> None:
        """Method to set endpoints for the quantum channel.

        This must be performed before transmission.

        Args:
            sender (Node): node sending qubits.
            receiver (str): name of node receiving qubits.
        """

        log.logger.info(
            "Set {} {} as ends of quantum channel {}".format(sender.name,
                                                             receiver,
                                                             self.name))
        self.sender = sender
        self.receiver = receiver
        sender.assign_qchannel(self, receiver)


    def transmit(self, qubit: "Photon", source: "Node") -> None:
        """Method to transmit photon-encoded qubits.

        Args:
            qubit (Photon): photon to be transmitted.
            source (Node): source node sending the qubit.

        Side Effects:
            Receiver node may receive the qubit (via the `receive_qubit` method).
        """

        log.logger.info(
            "{} send qubit with state {} to {} by Channel {}".format(
                self.sender.name, qubit.quantum_state, self.receiver,
                self.name))

        assert self.delay >= 0 and self.loss < 1, \
            "QuantumChannel init() function has not been run for {}".format(self.name)
        assert source == self.sender

        # remove lowest time bin
        if len(self.send_bins) > 0:
            time = -1
            while time < self.timeline.now():
                time_bin = hq.heappop(self.send_bins)
                time = int(time_bin * (1e12 / self.frequency))
            assert time == self.timeline.now(), "qc {} transmit method called at invalid time".format(self.name)

        # check if photon state using Fock representation
        if qubit.encoding_type["name"] == "fock":
            key = qubit.quantum_state  # if using Fock representation, the `quantum_state` field is the state key.
            # apply loss channel on photonic statex
            self.timeline.quantum_manager.add_loss(key, self.loss)

            # schedule receiving node to receive photon at future time determined by light speed
            future_time = self.timeline.now() + self.delay
            process = Process(self.receiver, "receive_qubit", [source.name, qubit])
            event = Event(future_time, process)
            self.timeline.schedule(event)

        # if not using Fock representation, check if photon kept
        elif (self.sender.get_generator().random() > self.loss) or qubit.is_null:
            if self._receiver_on_other_tl():
                self.timeline.quantum_manager.move_manage_to_server(
                    qubit.quantum_state)

            if qubit.is_null:
                qubit.add_loss(self.loss)

            # check if polarization encoding and apply necessary noise
            if (qubit.encoding_type["name"] == "polarization") and (
                    self.sender.get_generator().random() > self.polarization_fidelity):
                qubit.random_noise(self.get_generator())

            # schedule receiving node to receive photon at future time determined by light speed
            future_time = self.timeline.now() + self.delay
            process = Process(self.receiver, "receive_qubit", [source.name, qubit])
            event = Event(future_time, process)
            self.timeline.schedule(event)

        # if not using Fock representation, if photon lost, exit
        else:
            pass

    def schedule_transmit(self, min_time: int) -> int:
        """Method to schedule a time for photon transmission.

        Quantum Channels are limited by a frequency of transmission.
        This method returns the next available time for transmitting a photon.
        
        Args:
            min_time (int): minimum simulation time for transmission.

        Returns:
            int: simulation time for next available transmission window.
        """

        # TODO: move this to node?

        min_time = max(min_time, self.timeline.now())
        time_bin = min_time * (self.frequency / 1e12)
        if time_bin - int(time_bin) > 0.00001:
            time_bin = int(time_bin) + 1
        else:
            time_bin = int(time_bin)

        # find earliest available time bin
        while time_bin in self.send_bins:
            time_bin += 1
        hq.heappush(self.send_bins, time_bin)

        # calculate time
        time = int(time_bin * (1e12 / self.frequency))
        return time

    def _receiver_on_other_tl(self) -> bool:
        return self.timeline.get_entity_by_name(self.receiver) is None


class ClassicalChannel(OpticalChannel):
    """Optical channel for transmission of classical messages.

    Classical message transmission is assumed to be lossless.

    Attributes:
        name (str): label for channel instance.
        timeline (Timeline): timeline for simulation.
        sender (Node): node at sending end of optical channel.
        receiver (Node): node at receiving end of optical channel.
        distance (float): length of the fiber (in m).
        delay (float): delay (in ps) of message transmission (default distance / light_speed).
    """

    def __init__(self, name: str, timeline: "Timeline", distance: int, pulse_width, max_power, filter_bandwidth, delay=-1):
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
        self.pulse_width = pulse_width
        self.classical_commnication_rate = 1e9
        self.max_power = max_power
        self.filter_bandwidth = filter_bandwidth

    def set_ends(self, sender: "Node", receiver: str) -> None:
        """Method to set endpoints for the classical channel.

        This must be performed before transmission.

        Args:
            sender (Node): node sending classical messages.
            receiver (str): name of node receiving classical messges.
        """

        log.logger.info(
            "Set {} {} as ends of classical channel {}".format(sender.name,
                                                               receiver,
                                                               self.name))
        # These sender and receiver correspond to the source and receiver for the quantum channel.
        # This helps us find the direction in which we need to find the Raman Scattering (forward or backward). 
        self.sender = sender
        self.receiver = receiver
        sender.assign_cchannel(self, receiver)


    def transmit(self, message: "Message", source: "Node", priority: int) -> None:
        """Method to transmit classical messages.

        Args:
            message (Message): message to be transmitted.
            source (Node): node sending the message.
            priority (int): priority of transmitted message (to resolve message reception conflicts).

        Side Effects:
            Receiver node may receive the qubit (via the `receive_qubit` method).
        """

        log.logger.info(
            "{} send message {} to {} by Channel {}".format(self.sender.name,
                                                            message,
                                                            self.receiver,
                                                            self.name))
        assert source == self.sender

        future_time = round(self.timeline.now() + int(self.delay))
        process = Process(self.receiver, "receive_message", [source.name, message])
        event = Event(future_time, process, priority)
        self.timeline.schedule(event)


    def start_classical_communication(self):
        self.sender_index = re.findall(r'\d+', self.sender.name)[0]
        self.pcap = PcapNgReader("pcap_files/SeQUeNCe-0-%s.pcap" % (self.sender_index))
        self.get_classical_communication()

    # The NS3 implementation would simply give us the packet data that needs to be communicated between the nodes in the network. The routing may also happen at NS3.
    # However, the actual simulation still needs to happen in SeQUeNCe. So, we can 
    def get_classical_communication(self):
        bit_timing_list = [] 
        # self.packet_number += 1
        packet = next(self.pcap, False)
        if not packet:
            return 


        print("time at classical communication:", self.timeline.now())
        bits = BitArray(raw(packet))
        for bit_number, i in enumerate(bits):
            if i == 1:
                bit_timing_list.append(self.timeline.now() + bit_number*1e12/self.classical_commnication_rate) # This function must be called when the packet needs to be sent. 

        print("sender index:", self.sender_index, "packet dst", packet.dst)

        direction = (packet.dst == "192.168.1.%s" % (int(self.sender_index)+1)) # This could be handled more elegenatly by keeping PCAP file names as IP addresses
        self.sender.qchannels[self.receiver].transmit_classical_message(bit_timing_list, self.pulse_width, direction, self.max_power, self.filter_bandwidth)

        process = Process(self, "get_classical_communication", [])
        event = Event(self.timeline.now() + packet.len*1e12/self.classical_commnication_rate, process)
        self.timeline.schedule(event)


