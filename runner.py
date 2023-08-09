from src.kernel.timeline import Timeline
from src.components.optical_channel import ClassicalChannel, QuantumChannel
from src.kernel.event import Event
from src.kernel.process import Process
from src.topology.node import raman_receiver_node, raman_sender_node
from src.entanglement_management.raman_protocols import RamanTestReceiver
from matplotlib import pyplot as plt
from scipy.stats import norm
import traceback
from decimal import Decimal
import numpy as np
import os

def init_experiment_setup(params):
    sender = raman_sender_node("sender_0", params["tl"], params["num_iterations"], params["classical_power1"], params["narrow_band_filter_bandwidth"], params["quantum_channel_wavelength"], 
                               params["mean_photon_num"], params["is_distinguishable"], params["pulse_separation"], params["batch_size"], params["pulse_width"])
    signal_receiver = raman_receiver_node("signal_receiver_1", params["tl"], 'sender', params["collection_probability"], params["dark_count_rate"], params["dead_time"], params["time_resolution"])
    idler_receiver = raman_receiver_node("idler_receiver_2", params["tl"], 'sender', params["collection_probability"], params["dark_count_rate"], params["dead_time"], params["time_resolution"])

    receiver_protocol = RamanTestReceiver(signal_receiver, idler_receiver, 'sender', params["pulse_separation"])

    signal_receiver.attach_detector_to_receiver(receiver_protocol)
    idler_receiver.attach_detector_to_receiver(receiver_protocol)
    sender.attach_lightsource_to_receivers("signal_receiver_1", "idler_receiver_2")


    signal_channel = QuantumChannel("signal_channel", params["tl"], params["quantum_channel_attenuation"], params["classical_channel_attenuation"], params["distance"], params["raman_coefficient"], params["classical_communication_rate"], params["classical_power1"]/params["extinction_ratio"], params["classical_power1"], params["narrow_band_filter_bandwidth"],
                                    params["polarization_fidelity"], params["light_speed"], params["max_rate"], params["quantum_channel_wavelength"], params["classical_channel_wavelength"])
    idler_channel = QuantumChannel("idler_channel", params["tl"], params["quantum_channel_attenuation"], params["classical_channel_attenuation"], params["distance"], params["raman_coefficient"], params["classical_communication_rate"], params["classical_power1"]/params["extinction_ratio"], params["classical_power1"], params["narrow_band_filter_bandwidth"],
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
    "classical_communication_rate" : 1e7/1e12, # Classical communication rate in Picoseconds, i.e. B/ps. 1e7/1e12 is 10Mb/s in ps = 1e-5 b/ps

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
    "classical_power1" : 0.0003,
    "extinction_ratio" : 10,
    "narrow_band_filter_bandwidth" : 0.03,
}


def calculate_BER_amplified(power, classical_channel_attenuation, classical_channel_wavelength, distance, B, n_sp = 1,  G = 1000):
    h = 6.62607015 * 10**(-34)
    c = 3 * 10**8    
    P_n = n_sp*h * c / classical_channel_wavelength
    P = power * np.exp(-classical_channel_attenuation*distance)

    QFactor = np.sqrt(G*P)/(np.sqrt((G-1)*P_n*B))
    BER = 1-norm.cdf(QFactor)

    return BER


def calculate_BER_APD(power1, power0, classical_channel_attenuation, distance, B, T = 300, R_L = 100, F_n = 3, G_m = 10, K_a = 0.7, R = 1.25):
    """Parameters here are:
        T: Temperature (K)
        R_L: Load Resistance (Ohms)
        F_n: Noise figure of front end amplifier (dB)
        G_m: Avalance Multiplicative Gain (=1 for PIN detectors) (Ratio)
        K_a: Ionization Coefficient Ratio for the APD (Ratio)
        R: Responsivity of the APD (Amp/Watt)

        The function defines:
        k_b: Boltzmann coefficient
        F_a: Excess Noise factor of the APD
    """
    
    k_b = 1.38e-23
    e = 1.6e-19
    var_thermal = 2*k_b*T*F_n*B/R_L
    
    received_power = lambda power: power*np.exp(-classical_channel_attenuation*distance) 

    current1 = G_m * R * received_power(power1)
    current0 = G_m * R * received_power(power0)

    F_A = K_a*G_m + (1-K_a)*(2-1/G_m)
    var_shot = lambda current: e*G_m*F_A*current*B

    sigma0 = np.sqrt(var_shot(current0) + var_thermal)
    sigma1 = np.sqrt(var_shot(current1) + var_thermal)

    QFactor = (current1-current0)/(sigma1+sigma0)
    print("Qfactor:", QFactor)

    return 1-norm.cdf(QFactor)


import math 
def calculate_BER_4PAM_noPreamplifier(classical_channel_attenuation, distance, B, OMA, avg_power = 4e-3, T = 298, R_L = 50, F_n = 5, RIN = -155, R = 0.4):
    delta_f = B/2 # Receiver Bandwidth
    k_b = 1.38e-23
    e = 1.6e-19
    RIN = 10**(RIN/10) # We receive the RIN in dB
    OMA = 10**(OMA/10)/1000 # We receive the OMA in dBm
    assert avg_power - OMA/2 > 0
    received_power = lambda power: power*np.exp(-classical_channel_attenuation*distance) 
    print("received power:", 10*np.log10(received_power(avg_power*1e3)))
    current = lambda power: R * received_power(power)
    thermal_noise = 4*k_b*T*F_n*delta_f/R_L
    # print("thermal_noise:", thermal_noise)
    # shot_noise = 2*e*current*delta_f 
    # RIN_noise = RIN*current**2*delta_f
    level_noise = lambda current: thermal_noise + 2*e*current*delta_f + RIN*(current**2)*delta_f
    # level_noise = lambda current: RIN*(current**2)*delta_f
    power_levels = [avg_power-OMA/2, avg_power-OMA/6, avg_power+OMA/6, avg_power+OMA/2]
    total_noise = sum([level_noise( current(power) ) for power in power_levels])
    rms_noise = np.sqrt(total_noise)
    print("total_noise:", total_noise, "RMS noise:", rms_noise, "avg_current:", current(avg_power), "parameter:", current(avg_power)/(3*np.sqrt(2)*rms_noise))
    SER = (0.75)*(math.erfc(current(avg_power)/(3*np.sqrt(2)*rms_noise)))
    hamming_distance_natural = 4/3
    BER = hamming_distance_natural * SER/np.log2(4)
    return BER

calculate_BER_4PAM_noPreamplifier(0.55, 2, 1.25e9, OMA = 0.5, avg_power = 1.5e-3)


# BER = calculate_BER_amplified(10**(i/10) * 1e-3,  experimental_parameters["classical_channel_attenuation"], experimental_parameters["classical_channel_wavelength"], experimental_parameters["distance"]*10, experimental_parameters["classical_channel_rate"]*1e12)
# os.system(f'cp NS3_script.cc /home/asingal/ns-3-allinone/ns-3-dev/scratch/SeQUeNCe.cc && /home/asingal/ns-3-allinone/ns-3-dev/ns3 run "SeQUeNCe.cc {i}"')

num_repetitions = 1
num_samples = 1
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