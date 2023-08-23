from src.kernel.timeline import Timeline
from src.components.optical_channel import ClassicalChannel, PULSE_QuantumChannel
from src.kernel.event import Event
from src.kernel.process import Process
from src.topology.node import raman_receiver_node, raman_sender_node
from src.topology.node import PULSE_BSMNode, raman_sender_node
from src.entanglement_management.raman_protocols import RamanTestReceiver
from matplotlib import pyplot as plt
import traceback
import numpy as np
import os
from datetime import datetime

def init_experiment_setup(params):
    sender1 = raman_sender_node("PULSE_Quantum_Router_0", params["tl"], params["num_iterations"], params["clock_power"], params["narrow_band_filter_bandwidth"], params["quantum_channel_wavelength"], 
                               params["mean_photon_num"], params["is_distinguishable"], params["pulse_separation"], params["batch_size"], params["pulse_width"])
    sender2 = raman_sender_node("PULSE_Quantum_Router_2", params["tl"], params["num_iterations"], params["clock_power"], params["narrow_band_filter_bandwidth"], params["quantum_channel_wavelength"], 
                               params["mean_photon_num"], params["is_distinguishable"], params["pulse_separation"], params["batch_size"], params["pulse_width"])
    
    receiver = PULSE_BSMNode("PULSE_BSM_1", params["tl"], ['PULSE_Quantum_Router_0', "PULSE_Quantum_Router_2"])
    # idler_receiver = PULSE_BSM("idler_receiver", params["tl"], 'sender', params["collection_probability"], params["dark_count_rate"], params["dead_time"], params["time_resolution"])

    receiver_protocol = RamanTestReceiver(receiver, params["pulse_separation"])

    # signal_receiver.attach_detector_to_receiver(receiver_protocol)
    receiver.attach_detector_to_receiver(receiver_protocol)
    sender1.attach_lightsource_to_receivers("PULSE_BSM_1")
    sender2.attach_lightsource_to_receivers("PULSE_BSM_1")


    Qchannel1 = PULSE_QuantumChannel("Qchannel1", params["tl"], params["quantum_channel_attenuation"], params["classical_channel_attenuation"], params["distance"], params["raman_coefficient"], 
                                    params["polarization_fidelity"], params["light_speed"], params["max_rate"], params["quantum_channel_wavelength"], params["classical_channel_wavelength"], window_size = params["window_size"])
    Qchannel2 = PULSE_QuantumChannel("Qchannel2", params["tl"], params["quantum_channel_attenuation"], params["classical_channel_attenuation"], params["distance"], params["raman_coefficient"], 
                                    params["polarization_fidelity"], params["light_speed"], params["max_rate"], params["quantum_channel_wavelength"], params["classical_channel_wavelength"], window_size = params["window_size"])
    Cchannel1 = ClassicalChannel("Cchannel1", params["tl"], params["distance"], params["pulse_width"], params["clock_power"], params["narrow_band_filter_bandwidth"])
    Cchannel2 = ClassicalChannel("Cchannel2", params["tl"], params["distance"], params["pulse_width"], params["clock_power"], params["narrow_band_filter_bandwidth"])

    Qchannel1.set_ends(sender1, "PULSE_BSM_1")
    Qchannel2.set_ends(sender2, "PULSE_BSM_1")
    Cchannel1.set_ends(sender1, "PULSE_BSM_1")
    Cchannel2.set_ends(sender2, "PULSE_BSM_1")

    return sender1, sender2


experimental_parameters = {
    "tl" : Timeline(5000e12),

    # Parameters
    # Detector_parameters
    "collection_probability" : 10**(-1.2),
    "dark_count_rate" : 100, #100,
    "dead_time" : 25000,
    "time_resolution" : 50,

    # Optical channel
    "quantum_channel_attenuation" : 0.44,
    "classical_channel_attenuation" : 0.5,
    "distance" : 2,
    "raman_coefficient" : 33e-10,
    "polarization_fidelity" : 1,
    "light_speed" : 3e8,
    "max_rate" : 1e12,
    "quantum_channel_wavelength" : 1536e-9,
    "classical_channel_wavelength" : 1610e-9,
    "window_size" : 1e11,
    # clock parameters
    "pulse_width" : 1e3,
    "clock_power" : 0.0003,
    "narrow_band_filter_bandwidth" : 0.03,

    # Light Source
    # "wavelength" : quantum_channel_wavelength,
    "mean_photon_num" : 2, # 0.00316228, # 0.01
    "is_distinguishable" : True,
    "pulse_separation" : 5e3,
    "batch_size" : 5000, # 50000000,
    "num_iterations" : 5, # 240, 600,
}

os.system("cp ns3_script.cc /home/anshsingal/ns-allinone-3.38/ns-3.38/scratch/SeQUeNCe.cc && /home/anshsingal/ns-allinone-3.38/ns-3.38/ns3 run SeQUeNCe.cc")


try:
    now = datetime.now()
    print("time of start:", now)
    experimental_parameters["tl"] = Timeline(5000e12)
    # experimental_parameters["mean_photon_num"] = 0.01
    sender1, sender2 = init_experiment_setup(experimental_parameters)
    
    process1 = Process(sender1.protocol, "start", [True])
    event1 = Event(0, process1)
    experimental_parameters["tl"].schedule(event1)

    process2 = Process(sender2.protocol, "start", [False])
    event2 = Event(0, process2)
    experimental_parameters["tl"].schedule(event2)

    experimental_parameters["tl"].init()
    experimental_parameters["tl"].run()
    #     file = open("CAR_Data.txt")
    #     CAR_Data.append(list(map(float, file.readlines())))
    #     file.close()
    #     os.remove("CAR_Data.txt")
    # plt.boxplot(list(zip(*CAR_Data)))
    # plt.xticks(np.arange(num_samples)+1, np.logspace(min_mpn, max_mpn, num_samples))

        

    # print(signal_receiver.protocol.coincidence_times)
    
    # n, bins, patches = plt.hist(signal_receiver.protocol.detection_times, range(-28125, 28126, 6250))
    # plt.show() 
    print("simulation ended at", datetime.now())   
except Exception:
    print(traceback.format_exc())