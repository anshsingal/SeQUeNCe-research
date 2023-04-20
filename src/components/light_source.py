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

                # process0 = Process(self.direct_receiver, "get", [new_photon0])
                # process1 = Process(self.another_receiver, "get", [new_photon1])
                # event0 = Event(int(round(time)), process0)
                # event1 = Event(int(round(time)), process1)
                # self.timeline.schedule(event0)
                # self.timeline.schedule(event1)

                self.photon_counter += 1

            # time += 1e12 / self.frequency
        return out

    def assign_another_receiver(self, receiver):
        self.another_receiver = receiver



# class ParametricSource(Entity):
#     def __init__(self, own, name, timeline, signal_receiver, idler_receiver, wavelength,
#                  mean_photon_num, distinguishable, batch_size, pulse_separation, pulse_width = 50):
#         Entity.__init__(self, name, timeline)
#         self.own = own
#         self.wavelength = wavelength
#         self.mean_photon_num = mean_photon_num
#         self.signal_receiver = signal_receiver
#         self.idler_receiver = idler_receiver
#         self.distinguishable = distinguishable 
#         self.pulse_width = pulse_width # width of individual temporal mode in ps
#         self.pulse_separation = pulse_separation
#         self.transmit_time = self.own.timeline.now()
#         self.batch_size  = batch_size
#         # self.wavelengths = wavelengths

#     def init(self):
#         pass

#     def emit(self):
#         # print("light source emission called")
#         """Method to emit photons.

#         Will emit photons for a length of time determined by the `state_list` parameter.
#         The number of photons emitted per period is calculated as a poisson random variable.

#         Arguments:
#             state_list (List[List[complex]]): list of complex coefficient arrays to send as photon-encoded qubits.
#         """

#         # if self.distinguishable == True:
#         #     num_photon_pairs = self.get_generator().poisson(self.mean_photon_num)
#         # else:
#             # << Implement Thermal distribution. >> 
#         # signal_pulse = [0]*num_photon_pairs
#         # idler_pulse = [0]*num_photon_pairs
        
#         num_photon_pairs = np.random.poisson(self.mean_photon_num, (self.batch_size, 1))
#         if num_photon_pairs == 0:
#             self.transmit_time = max(self.transmit_time, self.timeline.now()) + self.pulse_separation + self.pulse_width
#             # print("No photons emitted")
#             return
#         last_arrival = self.pulse_width + 1
#         while last_arrival > self.pulse_width and last_arrival <= 0:
#             last_arrival = np.random.gamma(shape = num_photon_pairs, scale = 1/self.mean_photon_num)
#         # print("last_arrival:", last_arrival)

#         arrival_times = np.append(np.random.randint(0, last_arrival, num_photon_pairs-1), [int(last_arrival)]) 

        
#         # print("Light source emission time:", arrival_times)

#         signal_emit_time = None
#         idler_emit_time = None

#         for pulse_position in arrival_times:
#             state = np.around(np.random.rand())
#             new_photon0 = Photon(None, wavelength=self.wavelength, quantum_state=(complex(state), complex(1-state)))
#             new_photon1 = Photon(None, wavelength=self.wavelength, quantum_state=(complex(state), complex(1-state)))

#             self.transmit_time = max(self.transmit_time, self.timeline.now()) + self.pulse_separation + pulse_position

#             signal_photon = new_photon0
#             idler_photon = new_photon1

#             signal_emit_time = self.own.schedule_qubit(self.signal_receiver, self.transmit_time)
#             idler_emit_time = self.own.schedule_qubit(self.idler_receiver, self.transmit_time)

#             assert signal_emit_time == idler_emit_time

#             # print(f"emission times: {signal_emit_time}")

#             signal_process = Process(self.own, "send_qubit", [self.signal_receiver, signal_photon])
#             idler_process = Process(self.own, "send_qubit", [self.idler_receiver, idler_photon])
            
#             signal_event = Event(signal_emit_time, signal_process)
#             idler_event = Event(idler_emit_time, idler_process)
            
#             self.own.timeline.schedule(signal_event)
#             self.own.timeline.schedule(idler_event)
        
#         if signal_emit_time:
#             return signal_emit_time
            


#     def assign_another_receiver(self, receiver):
#         self.another_receiver = receiver




class ParametricSource(Entity):
    def __init__(self, own, name, timeline, signal_receiver, idler_receiver, wavelength,
                 mean_photon_num, is_distinguishable, pulse_separation, batch_size, pulse_width = 50):
        Entity.__init__(self, name, timeline)

        self.own = own
        self.wavelength = wavelength
        self.mean_photon_num = mean_photon_num
        self.signal_receiver = signal_receiver
        self.idler_receiver = idler_receiver
        # self.distinguishable = distinguishable 
        self.pulse_width = pulse_width # width of individual temporal mode in ps
        self.pulse_separation = pulse_separation
        self.transmit_time = self.own.timeline.now()
        self.batch_size  = batch_size
        self.pulse_window_ID = 0
        # self.wavelengths = wavelengths

    def init(self):
        pass

    def schedule_emit(self):
        # print("batch length:", self.batch_size * (self.pulse_separation + self.pulse_width))
        # print("light source emission called")
        """Method to emit photons.

        Will emit photons for a length of time determined by the `state_list` parameter.
        The number of photons emitted per period is calculated as a poisson random variable.

        Arguments:
            state_list (List[List[complex]]): list of complex coefficient arrays to send as photon-encoded qubits.
        """

        # if self.distinguishable == True:
        #     num_photon_pairs = self.get_generator().poisson(self.mean_photon_num)
        # else:
            # << Implement Thermal distribution. >> 
        # signal_pulse = [0]*num_photon_pairs
        # idler_pulse = [0]*num_photon_pairs
        
        
        # print("num_photon_pairs produced:",  num_photon_pairs)
        # if num_photon_pairs == 0:
        #     self.transmit_time = max(self.transmit_time, self.timeline.now()) + self.pulse_separation + self.pulse_width
        #     # print("No photons emitted")
        #     return

        train_duration = (self.batch_size+1) * (self.pulse_width + self.pulse_separation)

        signal_emit_time = self.own.schedule_qubit(self.signal_receiver, train_duration)
        idler_emit_time = self.own.schedule_qubit(self.idler_receiver, train_duration)

        assert signal_emit_time == idler_emit_time


        # for i,j in zip(signal_pulse_train.photon_counts,self.own.timeline.now() + signal_pulse_train.time_offsets):
        #     print("photons sent:", i, "at", j)
        

        process = Process(self, "emit", [])
        event = Event(signal_emit_time, process)
        self.own.timeline.schedule(event)
        

        # print("Sent idler photon train:")
        # for i,j in zip(idler_pulse_train.photon_counts,idler_pulse_train.time_offsets):
        #     print(i, j)

        # print("Sent signal photon train:")
        # for i,j in zip(signal_pulse_train.photon_counts,signal_pulse_train.time_offsets):
        #     print(i, j)

        # print("num sent signal photon train:", len(signal_pulse_train.photon_counts), "at time:", signal_emit_time)
        
        # signal_process = Process(self.own, "send_qubit", [self.signal_receiver, signal_pulse_window])
        # idler_process = Process(self.own, "send_qubit", [self.idler_receiver, idler_pulse_window])
        
        # signal_event = Event(signal_emit_time, signal_process)
        # idler_event = Event(idler_emit_time, idler_process)
        
        # self.own.timeline.schedule(signal_event)
        # self.own.timeline.schedule(idler_event)
        # print("emission events scheduled")

        return signal_emit_time + self.batch_size * (self.pulse_separation + self.pulse_width)

    def emit(self):
        num_photon_pairs = np.random.poisson(self.mean_photon_num, (int(self.batch_size), 1))
        signal_pulse_train = PulseTrain(num_photon_pairs, self.pulse_width, self.pulse_separation, self.wavelength)
        idler_pulse_train = copy(signal_pulse_train)
        # print("initial time offsets", signal_pulse_train.time_offsets)

        signal_pulse_window = PulseWindow(self.pulse_window_ID)
        idler_pulse_window = PulseWindow(self.pulse_window_ID)
        self.pulse_window_ID += 1 

        signal_pulse_window.source_train.append(signal_pulse_train)
        idler_pulse_window.source_train.append(idler_pulse_train)

        self.own.send_qubit(self.signal_receiver, signal_pulse_window)
        self.own.send_qubit(self.idler_receiver, idler_pulse_window)

        # for i in num_photon_pairs:
                
        #     last_arrival = self.pulse_width + 1
        #     while last_arrival > self.pulse_width and last_arrival <= 0:
        #         last_arrival = np.random.gamma(shape = num_photon_pairs, scale = 1/self.mean_photon_num)
        #     # print("last_arrival:", last_arrival)

        #     arrival_times = np.append(np.random.randint(0, last_arrival, num_photon_pairs-1), [int(last_arrival)]) 

        
        # print("Light source emission time:", arrival_times)

        # signal_emit_time = None
        # idler_emit_time = None

        # for pulse_position in arrival_times:
        #     state = np.around(np.random.rand())
        #     new_photon0 = Photon(None, wavelength=self.wavelength, quantum_state=(complex(state), complex(1-state)))
        #     new_photon1 = Photon(None, wavelength=self.wavelength, quantum_state=(complex(state), complex(1-state)))

        #     self.transmit_time = max(self.transmit_time, self.timeline.now()) + self.pulse_separation + pulse_position

        #     signal_photon = new_photon0
        #     idler_photon = new_photon1

        #     signal_emit_time = self.own.schedule_qubit(self.signal_receiver, self.transmit_time)
        #     idler_emit_time = self.own.schedule_qubit(self.idler_receiver, self.transmit_time)

        #     assert signal_emit_time == idler_emit_time

        #     # print(f"emission times: {signal_emit_time}")

        #     signal_process = Process(self.own, "send_qubit", [self.signal_receiver, signal_photon])
        #     idler_process = Process(self.own, "send_qubit", [self.idler_receiver, idler_photon])
            
        #     signal_event = Event(signal_emit_time, signal_process)
        #     idler_event = Event(idler_emit_time, idler_process)
            
        #     self.own.timeline.schedule(signal_event)
        #     self.own.timeline.schedule(idler_event)
        
        # if signal_emit_time:
        #     return signal_emit_time
            


    
    # def emit(self):
    #     """Method to emit photons.

    #     Will emit photons for a length of time determined by the `state_list` parameter.
    #     The number of photons emitted per period is calculated as a poisson random variable.

    #     Arguments:
    #         state_list (List[List[complex]]): list of complex coefficient arrays to send as photon-encoded qubits.
    #     """

    #     if self.distinguishable == True:
    #         num_photon_pairs = self.get_generator().poisson(self.mean_photon_num)
    #     # else:
    #         # << Implement Thermal distribution. >> 
    #     signal_pulse = [0]*num_photon_pairs
    #     idler_pulse = [0]*num_photon_pairs

    #     for i in range(num_photon_pairs):
    #         state = np.around(np.random.rand())
    #         new_photon0 = Photon(None, wavelength=self.wavelength, quantum_state=(complex(state), complex(1-state)))
    #         new_photon1 = Photon(None, wavelength=self.wavelength, quantum_state=(complex(state), complex(1-state)))

    #         intermodal_space = 50 + (np.random.rand()*50)

    #         signal_pulse[i] = (new_photon0, intermodal_space)
    #         idler_pulse[i] = (new_photon1, intermodal_space)

    #     signal_emit_time = self.own.schedule_qubit(self.signal_receiver, 0)
    #     idler_emit_time = self.own.schedule_qubit(self.idler_receiver, 0)

    #     assert signal_emit_time == idler_emit_time

    #     signal_process = Process(self.own, "send_qubit", [self.signal_receiver, signal_pulse])
    #     idler_process = Process(self.own, "send_qubit", [self.idler_receiver, idler_pulse])
        
    #     signal_event = Event(signal_emit_time, signal_process)
    #     idler_event = Event(idler_emit_time, idler_process)
        
    #     self.own.timeline.schedule(signal_event)
    #     self.own.timeline.schedule(idler_event)



    def assign_another_receiver(self, receiver):
        self.another_receiver = receiver
