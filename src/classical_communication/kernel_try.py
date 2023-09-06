import cupy as cp
from bitstring import BitArray
import random
import time

experimental_parameters = {
    # Detector_parameters
    "collection_probability" : 10**(-1.2),
    "dark_count_rate" : 100, #100,
    "dead_time" : 25000,
    "time_resolution" : 50,

    # Optical channel
    "quantum_channel_attenuation" : 0.44,
    "classical_channel_attenuation" : 0.55,
    "distance" : 5.,
    "raman_coefficient" : 10.5e-10,
    "polarization_fidelity" : 1,
    "light_speed" : 3e8,
    "max_rate" : 1e12,
    "quantum_channel_wavelength" : 1536e-9,
    "classical_channel_wavelength" : 1310e-9,
    "classical_communication_rate" : 1e7/1e12, # Classical communication (bit) rate in Picoseconds, i.e. B/ps. 1e7/1e12 is 10Mb/s in ps = 1e-5 b/ps
    "quantum_channel_index": 1.470,
    "classical_channel_index": 1.471,

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
OMA = 10**( experimental_parameters["OMA"] /10)/1000 # We receive the OMA in dBm
assert experimental_parameters["avg_power"] - OMA/2 > 0
experimental_parameters["classical_powers"] = [experimental_parameters["avg_power"]-OMA/2, experimental_parameters["avg_power"]-OMA/6, experimental_parameters["avg_power"]+OMA/6, experimental_parameters["avg_power"]+OMA/2]



kernel_file = open("classical_communication_kernel.cu", "r")
cuda_kernel = r"{}".format(kernel_file.read())

Raman_Kernel_call = cp.RawKernel( cuda_kernel, "Raman_Kernel" )

max_raman_photons_per_pulse = 5

s = BitArray([random.randrange(2) for _ in range(int(6))])

bits = cp.array(s, dtype = cp.bool_)
directions = cp.zeros(len(s), dtype = cp.bool_)
print(directions)
limit = int(len(bits)/2)
# print("input is:", list(map(int, bits)))
print("This is Python: len(bits):", len(bits))

print("sending initial data")
noise_photons = cp.zeros((limit, max_raman_photons_per_pulse), dtype = cp.int64) # We don't know how many photons could be generated per pulse. Taking 5 per bit (on average) for now arbitrarily
experimental_parameters["classical_powers"] = cp.array(experimental_parameters["classical_powers"], dtype = cp.float64)
print("sent data data")

h = 6.62607015 * 10**(-34)
c = 3. * 10**8
params = (bits, directions, noise_photons, limit, 
          experimental_parameters["classical_powers"], # Change this!
          experimental_parameters["raman_coefficient"],
          experimental_parameters["narrow_band_filter_bandwidth"],
          experimental_parameters["quantum_channel_attenuation"], # Change this!
          c/(experimental_parameters["classical_communication_rate"]*1e12)/1e3,
          experimental_parameters["classical_channel_attenuation"], # Change this!
          experimental_parameters["distance"] * 1000 / c,
          h, c, 
          experimental_parameters["quantum_channel_wavelength"],
          experimental_parameters["classical_communication_rate"],
          experimental_parameters["distance"],
          experimental_parameters["collection_probability"], # Change this!
          experimental_parameters["quantum_channel_index"],
          experimental_parameters["classical_channel_index"],
          max_raman_photons_per_pulse)

print("executing kernel")
start = time.time()

num_trials = 10000
num_photons = 0
for _ in range(num_trials):
    Raman_Kernel_call((limit//128+1,), (128,), params)
    cp.cuda.Stream.null.synchronize()

    for pulse in cp.asnumpy(noise_photons):
        for detection_time in pulse:
            if detection_time == 0:
                break
            num_photons += 1
    
print("required time:", time.time()-start)

print("Average number of Raman photons:", num_photons/num_trials)

# num_photons = 0

print(noise_photons)


# for pulse in cp.asnumpy(noise_photons):
#     for detection_time in pulse:
#         if detection_time == 0:
#             break
#         num_photons += 1
# print("num_photons:", num_photons)






# start = time.time()
# Raman_Kernel_call((limit//128+1,), (128,), params)
# cp.cuda.Stream.null.synchronize()
# print("required time:", time.time()-start)

# start = time.time()
# Raman_Kernel_call((limit//128+1,), (128,), params)
# cp.cuda.Stream.null.synchronize()
# print("required time:", time.time()-start)

# print("python now: classical_power:", experimental_parameters["classical_powers"])
# print("Pytho now: final result:\n", noise_photons)




# print("this is python: pulse_width:", c/(experimental_parameters["classical_communication_rate"]*1e12)/1e3)
# print("Python here: classical_powers", experimental_parameters["classical_powers"])

# import numpy as np
# raman_power = (experimental_parameters["classical_powers"][2] * experimental_parameters["raman_coefficient"] * experimental_parameters["narrow_band_filter_bandwidth"] * (np.exp(-experimental_parameters["quantum_channel_attenuation"] * c/(experimental_parameters["classical_communication_rate"]*1e12)/1e3) - np.exp(-experimental_parameters["classical_channel_attenuation"] * c/(experimental_parameters["classical_communication_rate"]*1e12)/1e3)) / (experimental_parameters["classical_channel_attenuation"] - experimental_parameters["quantum_channel_attenuation"]))
# print("Python raman power is:", raman_power)

# b