from src.kernel.timeline import Timeline
from src.components.optical_channel import ClassicalChannel, QuantumChannel
from src.kernel.event import Event
from src.kernel.process import Process
from src.topology.node import raman_receiver_node, raman_sender_node
from src.entanglement_management.raman_protocols import RamanTestReceiver
from src.classical_communication.BER_models import *
from matplotlib import pyplot as plt

import traceback
from decimal import Decimal
import numpy as np
import os

def init_experiment_setup(params):
    OMA = 10**( experimental_parameters["OMA"] /10)/1000 # We receive the OMA in dBm
    assert experimental_parameters["avg_power"] - OMA/2 > 0
    experimental_parameters["classical_powers"] = [experimental_parameters["avg_power"]-OMA/2, experimental_parameters["avg_power"]-OMA/6, experimental_parameters["avg_power"]+OMA/6, experimental_parameters["avg_power"]+OMA/2]

    sender = raman_sender_node("sender_0", params["tl"], params["num_iterations"], params["narrow_band_filter_bandwidth"], params["quantum_channel_wavelength"], 
                               params["mean_photon_num"], params["is_distinguishable"], params["pulse_separation"], params["batch_size"], params["pulse_width"])
    signal_receiver = raman_receiver_node("signal_receiver_1", params["tl"], 'sender', params["collection_probability"], params["dark_count_rate"], params["dead_time"], params["time_resolution"])
    idler_receiver = raman_receiver_node("idler_receiver_2", params["tl"], 'sender', params["collection_probability"], params["dark_count_rate"], params["dead_time"], params["time_resolution"])

    receiver_protocol = RamanTestReceiver(signal_receiver, idler_receiver, 'sender', params["pulse_separation"])

    signal_receiver.attach_detector_to_receiver(receiver_protocol)
    idler_receiver.attach_detector_to_receiver(receiver_protocol)
    sender.attach_lightsource_to_receivers("signal_receiver_1", "idler_receiver_2")


    signal_channel = QuantumChannel("signal_channel", params["tl"], params["quantum_channel_attenuation"], params["classical_channel_attenuation"], params["distance"], params["raman_coefficient"], params["classical_communication_rate"], experimental_parameters["classical_powers"], params["narrow_band_filter_bandwidth"],
                                    params["polarization_fidelity"], params["light_speed"], params["max_rate"], params["quantum_channel_wavelength"], params["classical_channel_wavelength"])
    idler_channel = QuantumChannel("idler_channel", params["tl"], params["quantum_channel_attenuation"], params["classical_channel_attenuation"], params["distance"], params["raman_coefficient"], params["classical_communication_rate"], experimental_parameters["classical_powers"], params["narrow_band_filter_bandwidth"],
                                    params["polarization_fidelity"], params["light_speed"], params["max_rate"], params["quantum_channel_wavelength"], params["classical_channel_wavelength"])
    classical_channel = ClassicalChannel("classical_channel", params["tl"], params["distance"], params["num_packets"], params["classical_communication_rate"])
    signal_channel.set_ends(sender, "signal_receiver_1")
    idler_channel.set_ends(sender, "idler_receiver_2")
    classical_channel.set_ends(sender, "signal_receiver_1")

    return sender



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
    "polarization_fidelity" : 1,
    "light_speed" : 3e8,
    "max_rate" : 1e12,
    "quantum_channel_wavelength" : 1536e-9,
    "classical_channel_wavelength" : 1610e-9,
    "classical_communication_rate" : 1e7/1e12, # Classical communication (bit) rate in Picoseconds, i.e. B/ps. 1e7/1e12 is 10Mb/s in ps = 1e-5 b/ps

    # Light Source
    # "wavelength" : quantum_channel_wavelength,
    "mean_photon_num" : 0.003, # 0.01
    "is_distinguishable" : True,
    "pulse_separation" : 5e3,
    "pulse_width" : 80,
    "batch_size" : 50000000,
    "num_iterations" : 10, # 240, 600,
    "num_packets" : 5,

    # Clock parameters
    "avg_power": 1.5e-3, # avg_power is written in W
    "OMA": 0.5, # OMA is written in dBm
    "narrow_band_filter_bandwidth" : 0.03,
}

# BER = calculate_BER_amplified(10**(i/10) * 1e-3,  experimental_parameters["classical_channel_attenuation"], experimental_parameters["classical_channel_wavelength"], experimental_parameters["distance"]*10, experimental_parameters["classical_channel_rate"]*1e12)
# os.system(f'cp NS3_script.cc /home/asingal/ns-3-allinone/ns-3-dev/scratch/SeQUeNCe.cc && /home/asingal/ns-3-allinone/ns-3-dev/ns3 run "SeQUeNCe.cc {i}"')

num_repetitions = 1 # 3
num_samples = 1 # 10
params = np.linspace(5, 6, num_samples)

try:
    CAR_Data = []
    for i in [5]: # Assuming power randes from -3 dBm (= 10^((-3)/10) mW) to -1 dBm (= 10^((-1)/10) mW)
            # NOTE Here that we are using the amplified BER for very small distances and hence multipled the distance by 10
        experimental_parameters["classical_communication_rate"] = 10**i / 1e12
        # ber = calculate_BER_APD(experimental_parameters["clock_power"], experimental_parameters["clock_power"]/10, experimental_parameters["classical_channel_attenuation"], experimental_parameters["distance"], experimental_parameters["classical_communication_rate"], G_m = 1)
        
        for j in range(num_repetitions):
            print("I,j:", i, j)
            experimental_parameters["tl"] = Timeline(5000e12)
            sender = init_experiment_setup(experimental_parameters)
            process = Process(sender.protocol, "start", [])
            event = Event(0, process)
            experimental_parameters["tl"].schedule(event)

            experimental_parameters["tl"].init()
            experimental_parameters["tl"].run()
        file = open("CAR_Data.txt")
        CAR_Data.append(list(map(float, file.readlines())))
        file.close()
   
except Exception:
    print(traceback.format_exc())



# CAR_Data = []
# temp_data = []
# file = open("CAR_Data.txt")
# temp_data.append(list(map(float, file.readlines())))
# file.close()
# print("temp_data:", temp_data)

# # print("len of temp_data:", len(temp_data[0]))

# # for i in range(3):
# #     CAR_Data.append(temp_data[0][i*num_trials:(i+1)*num_trials])

# for i in range(num_samples):
#     # print("limits:", i*num_repetitions, (i+1)*num_repetitions-1)
#     # print("car_dta:", temp_data[i*num_repetitions:(i+1)*num_repetitions-1])
#     CAR_Data.append(temp_data[0][i*num_repetitions:(i+1)*num_repetitions])

# print("CAR data:", CAR_Data)

# fig, ax = plt.subplots()
# fig.canvas.draw()
# labels = [str(np.log10(calculate_BER_APD(experimental_parameters["classical_power1"], experimental_parameters["classical_power1"]/experimental_parameters["extinction_ratio"], experimental_parameters["classical_channel_attenuation"], experimental_parameters["distance"], 10**i, G_m = 1)))[:4] for i in params]
# ax.set_xticklabels(labels)
# ax.set_xlabel("BER")
# ax.set_ylabel("CAR")
# plt.boxplot(CAR_Data, meanline = True, showmeans=True, medianprops={"linewidth":0})
# plt.savefig('PowerVsCAR.png')