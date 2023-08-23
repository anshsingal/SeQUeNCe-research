"""Models for simulation of photon emission devices.

This module defines the LightSource class to supply individual photons and the SPDCSource class to supply pre-entangled photons.
These classes should be connected to one or two entities, respectively, that are capable of receiving photons.
"""

from typing import List

from numpy import multiply, sqrt, zeros, kron, outer
import numpy as np

from .photon import Photon
from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process
from ..utils.encoding import polarization, fock
from ..utils import log
from ..utils.encoding import absorptive
from ..components.pulse_train import PulseTrain, PulseWindow


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

    def init(self):
        """Implementation of Entity interface (see base class)."""

        pass

    # for general use
    def emit(self, state_list) -> None:
        """Method to emit photons.

        Will emit photons for a length of time determined by the `state_list` parameter.
        The number of photons emitted per period is calculated as a poisson random variable.

        Arguments:
            state_list (List[List[complex]]): list of complex coefficient arrays to send as photon-encoded qubits.
        """

        log.logger.info("{} emitting {} photons".format(self.name, len(state_list)))

        time = self.timeline.now()
        period = int(round(1e12 / self.frequency))

        for i, state in enumerate(state_list):
            num_photons = self.get_generator().poisson(self.mean_photon_num)

            if self.get_generator().random() < self.phase_error:
                state = multiply([1, -1], state)

            for _ in range(num_photons):
                wavelength = self.linewidth * self.get_generator().standard_normal() + self.wavelength
                new_photon = Photon(str(i), self.timeline,
                                    wavelength=wavelength,
                                    location=self.owner,
                                    encoding_type=self.encoding_type,
                                    quantum_state=state)
                process = Process(self._receivers[0], "get", [new_photon])
                event = Event(time, process)
                self.timeline.schedule(event)
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
        wavelengths (List[float]): wavelengths (in nm) of emitted entangled photons.
            If a list is given, it should contain two elements (corresponding to two modes).
        linewidth (float): st. dev. in photon wavelength (in nm) (currently unused).
        mean_photon_num (float): mean number of photons emitted each period.
        encoding_type (Dict): encoding scheme of emitted photons (as defined in the encoding module).
        phase_error (float): phase error applied to qubits.
    """

    def __init__(self, name, timeline, wavelengths=None, frequency=8e7, mean_photon_num=0.1,
                 encoding_type=fock, phase_error=0, bandwidth=0):
        super().__init__(name, timeline, frequency, 0, bandwidth, mean_photon_num, encoding_type, phase_error)
        self.wavelengths = wavelengths
        # If the user uses a setting where both the photons have different wavelengths, you take the object 
        # from the user and use it directly. Otherwise, the direct method is to call set_wavelengths which sets 
        # both photons to have 1550nm wavelength. 
        if self.wavelengths is None or len(self.wavelengths) != 2:
            self.set_wavelength()

    def init(self):
        assert len(self._receivers) == 2, "SPDC source must connect to 2 receivers."

    def _generate_tmsv_state(self):
        """Method to generate two-mode squeezed vacuum state of two output photonic modes

        Returns:
            array: generated state.
        """

        mean_num = self.mean_photon_num
        truncation = self.timeline.quantum_manager.truncation

        # print("truncation is:", truncation)

        # create state component amplitudes list
        amp_list = [(sqrt(mean_num / (mean_num + 1)) ** m) / sqrt(mean_num + 1) for m in range(truncation)]
        amp_square_list = [amp ** 2 for amp in amp_list]
        amp_list.append(sqrt(1 - sum(amp_square_list)))

        # create two-mode state vector
        state_vec = zeros((truncation+1) ** 2)

        for i in range(truncation+1):
            amp = amp_list[i]
            basis = zeros(truncation+1)
            basis[i] = 1
            basis = kron(basis,basis)
            state_vec += amp * basis

        return state_vec

    def emit(self, state_list):
        """Method to emit photons.

        Will emit photons for a length of time determined by the `state_list` parameter.
        The number of photons emitted per period is calculated as a poisson random variable.

        Arguments:
            state_list (List[List[complex]]): list of complex coefficient arrays to send as photon-encoded qubits.
                This is ignored for absorptive and Fock encoding types.
                For these encoding types only the length of list matters and elements can be arbitrary.
        """

        log.logger.info("SPDC sourcee {} emitting {} photons".format(self.name, len(state_list)))

        time = self.timeline.now()

        if self.encoding_type["name"] == "fock":
            # Use Fock encoding.
            # The two generated photons should be entangled and should have keys pointing to same Fock state.
            for _ in state_list:
                # generate two new photons
                new_photon0 = Photon("", self.timeline,
                                     wavelength=self.wavelengths[0],
                                     location=self,
                                     encoding_type=self.encoding_type,
                                     use_qm=True)
                new_photon1 = Photon("", self.timeline,
                                     wavelength=self.wavelengths[1],
                                     location=self,
                                     encoding_type=self.encoding_type,
                                     use_qm=True)

                # set shared state to squeezed state
                state = self._generate_tmsv_state()
                keys = [new_photon0.quantum_state, new_photon1.quantum_state]
                self.timeline.quantum_manager.set(keys, state)

                self.send_photons(time, [new_photon0, new_photon1])
                self.photon_counter += 1
                time += 1e12 / self.frequency

        elif self.encoding_type["name"] == "absorptive":
            for _ in state_list:
                num_photon_pairs = self.get_generator().poisson(self.mean_photon_num)

                for _ in range(num_photon_pairs):
                    new_photon0 = Photon("", self.timeline,
                                         wavelength=self.wavelengths[0],
                                         location=self,
                                         encoding_type=self.encoding_type,
                                         use_qm=True)
                    new_photon1 = Photon("", self.timeline,
                                         wavelength=self.wavelengths[1],
                                         location=self,
                                         encoding_type=self.encoding_type,
                                         use_qm=True)

                    new_photon0.combine_state(new_photon1)
                    new_photon0.set_state((complex(0), complex(0), complex(0), complex(1)))
                    self.send_photons(time, [new_photon0, new_photon1])
                    self.photon_counter += 1

                if num_photon_pairs is 0:
                    # send two null photons for purposes of entanglement
                    new_photon0 = Photon("", self.timeline,
                                         wavelength=self.wavelengths[0],
                                         location=self,
                                         encoding_type=self.encoding_type,
                                         use_qm=True)
                    new_photon1 = Photon("", self.timeline,
                                         wavelength=self.wavelengths[1],
                                         location=self,
                                         encoding_type=self.encoding_type,
                                         use_qm=True)

                    new_photon0.is_null = True
                    new_photon1.is_null = True
                    new_photon0.combine_state(new_photon1)
                    new_photon0.set_state((complex(1), complex(0), complex(0), complex(0)))
                    self.send_photons(time, [new_photon0, new_photon1])

                time += 1e12 / self.frequency

        else:
            for state in state_list:
                num_photon_pairs = self.get_generator().poisson(
                self.mean_photon_num)

                if self.get_generator().random() < self.phase_error:
                    state = multiply([1, -1], state)

                for _ in range(num_photon_pairs):
                    new_photon0 = Photon("", self.timeline,
                                         wavelength=self.wavelengths[0],
                                         location=self,
                                         encoding_type=self.encoding_type)
                    new_photon1 = Photon("", self.timeline,
                                         wavelength=self.wavelengths[1],
                                         location=self,
                                         encoding_type=self.encoding_type)

                    new_photon0.combine_state(new_photon1)
                    new_photon0.set_state((state[0], complex(0), complex(0), state[1]))
                    self.send_photons(time, [new_photon0, new_photon1])
                    self.photon_counter += 1

                time += 1e12 / self.frequency

    def send_photons(self, time, photons: List["Photon"]):
        log.logger.debug("SPDC source {} sending photons to {} at time {}".format(
            self.name, self._receivers, time
        ))

        assert len(photons) == 2
        for dst, photon in zip(self._receivers, photons):
            # print("memory destintions", self._receivers)
            process = Process(dst, "get", [photon])
            event = Event(int(round(time)), process)
            self.timeline.schedule(event)

    def set_wavelength(self, wavelength1=1550, wavelength2=1550):
        """Method to set the wavelengths of photons emitted in two output modes."""
        self.wavelengths = [wavelength1, wavelength2]



class PULSE_ParametricSource(Entity):
    def __init__(self, own, name, timeline, receiver, wavelength,
                 mean_photon_num, is_distinguishable, pulse_separation, batch_size, pulse_width = 50):
        Entity.__init__(self, name, timeline)

        self.own = own
        self.wavelength = wavelength
        self.mean_photon_num = mean_photon_num
        self.receiver = receiver
        # self.idler_receiver = idler_receiver
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

        # train_duration = (self.batch_size+1) * (self.pulse_width + self.pulse_separation)

        # Schedules the qubit with the optical channel. 
        emit_time = self.own.schedule_qubit(self.receiver, self.timeline.now())

        print("emitting  now. Emission time is:", emit_time)
        # idler_emit_time = self.own.schedule_qubit(self.idler_receiver, train_duration)

        # assert signal_emit_time == idler_emit_time

        process = Process(self, "emit", [])
        event = Event(emit_time, process)
        self.own.timeline.schedule(event)
        
        return emit_time

    def emit(self):
        """Generates the pulse train based on the pulse separation, width and batch size. This method is called when the qubit can be accepted by the channel 

        Uses the Poisson process to find the number of photon pairs per pulse and create pulse trains which are stored in the pulse window. 

        """
        num_photon_pairs = np.random.poisson(self.mean_photon_num)
        # signal_pulse_train = PulseTrain(num_photon_pairs, self.pulse_width, self.pulse_separation, self.wavelength)
        # idler_pulse_train = signal_pulse_train.copy()

        photon = Photon(None, self.timeline, wavelength = self.wavelength, location = self.own.name, 
                        encoding_type=absorptive, quantum_state = num_photon_pairs, use_qm=True)
        
        print(f"Photon object created at {self.own.name}. sending over channel now. Photon number is:", photon.quantum_state)

        # signal_pulse_window = PulseWindow(self.pulse_window_ID)
        # idler_pulse_window = PulseWindow(self.pulse_window_ID)
        # self.pulse_window_ID += 1 

        # signal_pulse_window.source_train.append(signal_pulse_train)
        # idler_pulse_window.source_train.append(idler_pulse_train)

        # self.own.send_qubit(self.signal_receiver, signal_pulse_window)
        self.own.send_qubit(self.receiver, photon)


    def assign_another_receiver(self, receiver):
        self.another_receiver = receiver
