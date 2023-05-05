"""Models for simulation of photon emission devices.

This module defines the LightSource class to supply individual photons and the SPDCSource class to supply pre-entangled photons.
These classes should be connected to one or two entities, respectively, that are capable of receiving photons.
"""

from numpy import multiply
import numpy as np
from copy import copy

from .photon import Photon
from .pulse_train import PulseTrain, PulseWindow
from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process
from ..utils.encoding import polarization


class LightSource(Entity):
    """Model for a laser light source.

    The LightSource component acts as a simple low intensity laser, providing photon clusters at a set frequency.

    Attributes:
        name (str): label for beamsplitter instance
        timeline (Timeline): timeline for simulation
        frequency (float): frequency (in Hz) of photon creation.
        wavelength (float): wavelength (in nm) of emitted photons.
        linewidth (float): st. dev. in photon wavelength (in nm).
        mean_photon_num (float): mean number of photons emitted each period.
        encoding_type (Dict[str, Any]): encoding scheme of emitted photons (as defined in the encoding module).
        phase_error (float): phase error applied to qubits.
        photon_counter (int): counter for number of photons emitted.
    """

    def __init__(self, name, timeline, frequency=8e7, wavelength=1550, bandwidth=0, mean_photon_num=0.1,
                 encoding_type=polarization, phase_error=0):
        """Constructor for the LightSource class.

        Arguments:
            name (str): name of the light source instance.
            timeline (Timeline): simulation timeline.
            frequency (float): frequency (in Hz) of photon creation (default 8e7).
            wavelength (float): wavelength (in nm) of emitted photons (default 1550).
            bandwidth (float): st. dev. in photon wavelength (default 0).
            mean_photon_num (float): mean number of photons emitted each period (default 0.1).
            encoding_type (Dict): encoding scheme of emitted photons (as defined in the encoding module) (default polarization).
            phase_error (float): phase error applied to qubits (default 0).
        """

        Entity.__init__(self, name, timeline)
        self.frequency = frequency  # measured in Hz
        self.wavelength = wavelength  # measured in nm
        self.linewidth = bandwidth  # st. dev. in photon wavelength (nm)
        self.mean_photon_num = mean_photon_num
        self.encoding_type = encoding_type
        self.phase_error = phase_error
        self.photon_counter = 0
        # for BB84
        # self.basis_lists = []
        # self.basis_list = []
        # self.bit_lists = []
        # self.bit_list = []
        # self.is_on = False
        # self.pulse_id = 0

    def init(self):
        """Implementation of Entity interface (see base class)."""

        pass

    # for general use
    def emit(self, state_list, dst: str) -> None:
        """Method to emit photons.

        Will emit photons for a length of time determined by the `state_list` parameter.
        The number of photons emitted per period is calculated as a poisson random variable.

        Arguments:
            state_list (List[List[complex]]): list of complex coefficient arrays to send as photon-encoded qubits.
            dst (str): name of destination node to receive photons.
        """

        time = self.timeline.now()
        period = int(round(1e12 / self.frequency))

        for i, state in enumerate(state_list):
            num_photons = self.get_generator().poisson(self.mean_photon_num)

            if self.get_generator().random() < self.phase_error:
                state = multiply([1, -1], state)

            for _ in range(num_photons):
                wavelength = self.linewidth * self.get_generator().standard_normal() + self.wavelength
                new_photon = Photon(str(i),
                                    wavelength=wavelength,
                                    location=self.owner,
                                    encoding_type=self.encoding_type,
                                    quantum_state=state)
                process = Process(self.owner, "send_qubit", [dst, new_photon])
                event = Event(time, process)
                self.owner.timeline.schedule(event)
                self.photon_counter += 1
            time += period


class SPDCSource(LightSource):
    """Model for a laser light source for entangled photons (via SPDC).

    The SPDCLightSource component acts as a simple low intensity laser with an SPDC lens.
    It provides entangled photon clusters at a set frequency.

    Attributes:
        name (str): label for beamsplitter instance
        timeline (Timeline): timeline for simulation
        frequency (float): frequency (in Hz) of photon creation.
        wavelengths (float): wavelengths (in nm) of emitted entangled photons.
        linewidth (float): st. dev. in photon wavelength (in nm).
        mean_photon_num (float): mean number of photons emitted each period.
        encoding_type (Dict): encoding scheme of emitted photons (as defined in the encoding module).
        phase_error (float): phase error applied to qubits.
        photon_counter (int): counter for number of photons emitted.
        direct_receiver (Entity): device to receive one entangled photon.
        another_receiver (Entity): device to receive another entangled photon.
    """

    def __init__(self, name, timeline, direct_receiver=None, another_receiver=None, wavelengths=[], frequency=8e7, wavelength=1550,
                 bandwidth=0, mean_photon_num=0.1, encoding_type=polarization, phase_error=0):
        super().__init__(name, timeline, frequency, wavelength, bandwidth, mean_photon_num, encoding_type, phase_error)
        self.direct_receiver = direct_receiver
        self.another_receiver = another_receiver
        self.wavelengths = wavelengths

        

    def emit(self, state_list):
        """Method to emit photons.

        Will emit photons for a length of time determined by the `state_list` parameter.
        The number of photons emitted per period is calculated as a poisson random variable.

        Arguments:
            state_list (List[List[complex]]): list of complex coefficient arrays to send as photon-encoded qubits.
        """
        out = []
        # time = self.timeline.now()

        for state in state_list:
            num_photon_pairs = self.get_generator().poisson(self.mean_photon_num)
            print("num_photon_pairs:", num_photon_pairs)

            if self.get_generator().random() < self.phase_error:
                state = multiply([1, -1], state)

            for _ in range(num_photon_pairs):
                new_photon0 = Photon(None,
                                     wavelength=self.wavelengths[0],
                                     location=self.direct_receiver,
                                     encoding_type=self.encoding_type)
                new_photon1 = Photon(None,
                                     wavelength=self.wavelengths[1],
                                     location=self.direct_receiver,
                                     encoding_type=self.encoding_type)

                

                new_photon0.entangle(new_photon1)
                new_photon0.set_state((state[0], complex(0), complex(0), state[1]))

                out.append([new_photon0, new_photon1])

                self.photon_counter += 1

        return out

    def assign_another_receiver(self, receiver):
        self.another_receiver = receiver





class ParametricSource(Entity):
    def __init__(self, own, name, timeline, signal_receiver, idler_receiver, wavelength,
                 mean_photon_num, is_distinguishable, pulse_separation, batch_size, pulse_width = 50):
        Entity.__init__(self, name, timeline)

        self.own = own
        self.wavelength = wavelength
        self.mean_photon_num = mean_photon_num
        self.signal_receiver = signal_receiver
        self.idler_receiver = idler_receiver
        self.pulse_width = pulse_width # width of individual temporal mode in ps
        self.pulse_separation = pulse_separation
        self.transmit_time = self.own.timeline.now()
        self.batch_size  = batch_size
        self.pulse_window_ID = 0

    def init(self):
        pass

    def schedule_emit(self):
        """Method to schedule an event to start the emission process based on the time received from the optical channel.

        Will schedule an event to emit the photon train registerd with the optical channel. 

        """

        train_duration = (self.batch_size+1) * (self.pulse_width + self.pulse_separation)

        # Schedules the qubit with the optical channel. 
        signal_emit_time = self.own.schedule_qubit(self.signal_receiver, train_duration)
        idler_emit_time = self.own.schedule_qubit(self.idler_receiver, train_duration)

        assert signal_emit_time == idler_emit_time

        process = Process(self, "emit", [])
        event = Event(signal_emit_time, process)
        self.own.timeline.schedule(event)
        
        return signal_emit_time + self.batch_size * (self.pulse_separation + self.pulse_width)

    def emit(self):
        """Generates the pulse train based on the pulse separation, width and batch size. This method is called when the qubit can be accepted by the channel 

        Uses the Poisson process to find the number of photon pairs per pulse and create pulse trains which are stored in the pulse window. 

        """
        num_photon_pairs = np.random.poisson(self.mean_photon_num, (int(self.batch_size), 1))
        signal_pulse_train = PulseTrain(num_photon_pairs, self.pulse_width, self.pulse_separation, self.wavelength)
        idler_pulse_train = copy(signal_pulse_train)

        signal_pulse_window = PulseWindow(self.pulse_window_ID)
        idler_pulse_window = PulseWindow(self.pulse_window_ID)
        self.pulse_window_ID += 1 

        signal_pulse_window.source_train.append(signal_pulse_train)
        idler_pulse_window.source_train.append(idler_pulse_train)

        self.own.send_qubit(self.signal_receiver, signal_pulse_window)
        self.own.send_qubit(self.idler_receiver, idler_pulse_window)


    def assign_another_receiver(self, receiver):
        self.another_receiver = receiver
