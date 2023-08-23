from src.kernel.timeline import Timeline
from src.components.optical_channel import ClassicalChannel, QuantumChannel
from src.kernel.event import Event
from src.kernel.process import Process
from src.topology.node import raman_receiver_node, raman_sender_node
from src.topology.node import Raman_BSMNode, raman_sender_node
from src.entanglement_management.raman_protocols import RamanTestReceiver
from matplotlib import pyplot as plt
import traceback
import numpy as np
import os
from datetime import datetime

# def init_experiment_setup(params):
#     sender1 = raman_sender_node("Quantum_Router_0", params["tl"], params["num_iterations"], params["narrow_band_filter_bandwidth"], params["quantum_channel_wavelength"], 
#                                params["mean_photon_num"], params["is_distinguishable"], params["pulse_separation"], params["batch_size"], params["pulse_width"])
#     sender2 = raman_sender_node("Quantum_Router_2", params["tl"], params["num_iterations"], params["narrow_band_filter_bandwidth"], params["quantum_channel_wavelength"], 
#                                params["mean_photon_num"], params["is_distinguishable"], params["pulse_separation"], params["batch_size"], params["pulse_width"])
    
#     receiver = Raman_BSMNode("Raman_BSM_1", params["tl"], ['Quantum_Router_0', "Quantum_Router_2"])
#     # idler_receiver = PULSE_BSM("idler_receiver", params["tl"], 'sender', params["collection_probability"], params["dark_count_rate"], params["dead_time"], params["time_resolution"])

#     receiver_protocol = RamanTestReceiver(receiver, params["pulse_separation"])

#     # signal_receiver.attach_detector_to_receiver(receiver_protocol)
#     receiver.attach_detector_to_receiver(receiver_protocol)
#     sender1.attach_lightsource_to_receivers("Raman_BSM_1")
#     sender2.attach_lightsource_to_receivers("Raman_BSM_1")


#     Qchannel1 = PULSE_QuantumChannel("Qchannel1", params["tl"], params["quantum_channel_attenuation"], params["classical_channel_attenuation"], params["distance"], params["raman_coefficient"], 
#                                     params["polarization_fidelity"], params["light_speed"], params["max_rate"], params["quantum_channel_wavelength"], params["classical_channel_wavelength"], window_size = params["window_size"])
#     Qchannel2 = PULSE_QuantumChannel("Qchannel2", params["tl"], params["quantum_channel_attenuation"], params["classical_channel_attenuation"], params["distance"], params["raman_coefficient"], 
#                                     params["polarization_fidelity"], params["light_speed"], params["max_rate"], params["quantum_channel_wavelength"], params["classical_channel_wavelength"], window_size = params["window_size"])
#     Cchannel1 = ClassicalChannel("Cchannel1", params["tl"], params["distance"], params["pulse_width"], params["clock_power"], params["narrow_band_filter_bandwidth"])
#     Cchannel2 = ClassicalChannel("Cchannel2", params["tl"], params["distance"], params["pulse_width"], params["clock_power"], params["narrow_band_filter_bandwidth"])

#     Qchannel1.set_ends(sender1, "Raman_BSM_1")
#     Qchannel2.set_ends(sender2, "Raman_BSM_1")
#     Cchannel1.set_ends(sender1, "Raman_BSM_1")
#     Cchannel2.set_ends(sender2, "Raman_BSM_1")

#     return sender1, sender2


def init_experiment_setup(params):
    OMA = 10**( experimental_parameters["OMA"] /10)/1000 # We receive the OMA in dBm
    assert experimental_parameters["avg_power"] - OMA/2 > 0
    experimental_parameters["classical_powers"] = [experimental_parameters["avg_power"]-OMA/2, experimental_parameters["avg_power"]-OMA/6, experimental_parameters["avg_power"]+OMA/6, experimental_parameters["avg_power"]+OMA/2]

    sender0 = raman_sender_node("sender_0", params["tl"], params["num_iterations"], params["quantum_channel_wavelength"], params["mean_photon_num"], params["batch_size"])
    sender1 = raman_sender_node("sender_0", params["tl"], params["num_iterations"], params["quantum_channel_wavelength"], params["mean_photon_num"], params["batch_size"])

    receiver = Raman_BSMNode("Raman_BSM_1", params["tl"], ['Quantum_Router_0', "Quantum_Router_2"])
    receiver_protocol = RamanTestReceiver(receiver, params["pulse_separation"])

    receiver.attach_detector_to_receiver(receiver_protocol)
    sender0.attach_lightsource_to_receivers("Raman_BSM_1")
    sender1.attach_lightsource_to_receivers("Raman_BSM_1")

    qchannel0 = QuantumChannel("signal_channel", params)
    qchannel1 = QuantumChannel("idler_channel", params)
    cchannel0 = ClassicalChannel("classical_channel", params["tl"], params["distance"], params["classical_communication_rate"])
    cchannel1 = ClassicalChannel("classical_channel", params["tl"], params["distance"], params["classical_communication_rate"])
    
    qchannel0.set_ends(sender0, "Raman_BSM_1")
    qchannel1.set_ends(sender1, "Raman_BSM_1")
    cchannel0.set_ends(sender0, "Raman_BSM_1")
    cchannel1.set_ends(sender1, "Raman_BSM_1")

    return sender0, sender1



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
    "distance" : 5,
    "raman_coefficient" : 33e-10,
    "quantum_channel_wavelength" : 1536e-9,
    "classical_channel_wavelength" : 1610e-9,
    "classical_communication_rate" : 1e7/1e12, # Classical communication (bit) rate in Picoseconds, i.e. B/ps. 1e7/1e12 is 10Mb/s in ps = 1e-5 b/ps

    # Light Source
    # "wavelength" : quantum_channel_wavelength,
    "mean_photon_num" : 0.003, # 0.01
    "batch_size" : 50000000,
    "num_iterations" : 10, # 240, 600,

    # Clock parameters
    "avg_power": 1.5e-3, # avg_power is written in W
    "OMA": 3, # OMA is written in dBm
    "narrow_band_filter_bandwidth" : 0.03,
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