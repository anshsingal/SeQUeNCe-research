import time
import numpy as np
import cupy as cp


SPDC_FREQUENCY = 8e8
MODE_NUM = 10



params = {
    # quantum manager
    "TRUNCATION" : 1,  # truncation of Fock space (:dimension-1)

    # photon sources
    "TELECOM_WAVELENGTH" : 1436.,  # telecom band wavelength of SPDC source idler photon
    "QUANTUM_WAVELENGTH" : 1536.,  # wavelength of AFC memory resonant absorption, of SPDC source signal photon
    "SPDC_FREQUENCY" : SPDC_FREQUENCY,  # frequency of both SPDC sources' photon creation (same as memory frequency and detector count rate)
    "MEAN_PHOTON_NUM1" : 0.1,  # mean photon number of SPDC source on node 1
    "MEAN_PHOTON_NUM2" : 0.1,  # mean photon number of SPDC source on node 2

    # detectors
    "BSM_DET1_EFFICIENCY" : 0.15, # efficiency of detector 1 of BSM
    "BSM_DET2_EFFICIENCY" : 0.15,  # efficiency of detector 2 of BSM
    "BSM_DET1_DARK" : 0,  # Dark count rate (Hz)
    "BSM_DET2_DARK" : 0,
    "MEAS_DET1_EFFICIENCY" : 0.15,  # efficiency of detector 1 of DM measurement
    "MEAS_DET2_EFFICIENCY" : 0.15,  # efficiency of detector 2 of DM measurement
    "MEAS_DET1_DARK" : 0,
    "MEAS_DET2_DARK" : 0,
    "NBF_BANDWIDTH" : 0.03,

    # fibers
    "DIST_ANL_ERC" : 10.,  # distance between ANL and ERC, in km
    "DIST_HC_ERC" : 10.,  # distance between HC and ERC, in km
    "QUNATUM_ATTENUATION" : 0.076  * 10/np.log(10),  # attenuation rate of optical fibre (in dB/km)
    "DELAY_CLASSICAL" : 5e-3,  # delay for classical communication between BSM node and memory nodes (in s)
    "QUANTUM_INDEX" : 1.471,
    
    # L-Band
    "CLASSICAL_ATTENUATION" : [0.084 * 10/np.log(10), 0.084 * 10/np.log(10)], # attenuation rate of optical fibre (in dB/km)
    "RAMAN_COEFFICIENTS" : [33.e-10, 33.e-10], # Correct this!!!
    "CLASSICAL_INDEX" : [1.47, 1.47],
    "CLASSICAL_WAVELENGTH" : 1610,

    "DIRECTION": True, # True is co-propagation

    # Classical communication
    "CLASSICAL_RATE" : 1e10,
    "COMMS_WINDOW_SIZE" : (MODE_NUM/SPDC_FREQUENCY)*1e12, # Amount of time it takes to complete one batch of MODE_NUM quantum emissions. 
    "TOTAL_COMM_SIZE": (MODE_NUM/SPDC_FREQUENCY)*1e12, # This is the amount of time that we calculate Raman scatting for. For this simulation, these both are equal, i.e., we run one window for the entire duration. 
    "AVG_POWER": 10, # Here, you input a list of average powers (dBm, check this) that you want to iterate over. When the results are printed out, they'll contain only the avg_power that was used in that iteration. 
    # "OMA": 2, # Set later on in the program as a function of the AVG_POWER
    "MODULATION": "PAM4", 

    # memories
    "MODE_NUM" : MODE_NUM,  # number of temporal modes of AFC memory (same for both memories)
    "MEMO_FREQUENCY1" : SPDC_FREQUENCY,  # frequency of memory 1
    "MEMO_FREQUENCY2" : SPDC_FREQUENCY,  # frequency of memory 2
    "ABS_EFFICIENCY1" : 0.7,  # absorption efficiency of AFC memory 1
    "ABS_EFFICIENCY2" : 0.7,  # absorption efficiency of AFC memory 2
    "PREPARE_TIME1" : 0,  # time required for AFC structure preparation of memory 1
    "PREPARE_TIME2" : 0, # time required for AFC structure preparation of memory 2
    "COHERENCE_TIME1" : -1,  # spin coherence time for AFC memory 1 spinwave storage, -1 means infinite time
    "COHERENCE_TIME2" : -1,  # spin coherence time for AFC memory 2 spinwave storage, -1 means infinite time
    "AFC_LIFETIME1" : -1,  # AFC structure lifetime of memory 1, -1 means infinite time
    "AFC_LIFETIME2" : -1,  # AFC structure lifetime of memory 2, -1 means infinite time
    "DECAY_RATE1" : 4.3e-8,  # retrieval efficiency decay rate for memory 1
    "DECAY_RATE2" : 4.3e-8,  # retrieval efficiency decay rate for memory 2

    # experiment settings
    "time" : int(1e12),
    "calculate_fidelity_direct" : True,
    "calculate_rate_direct" : True,
    "num_direct_trials" : 3000,
    "num_bs_trials_per_phase" : 1,
    "phase_settings" : list(np.linspace(0, 2*np.pi, num=10, endpoint=False)),
}


def Raman_Kernel(symbol_number, direction, classical_power, 
raman_coefficient, narrow_band_filter_bandwidth, quantum_channel_attenuation, pulse_width,
classical_channel_attenuation, h, c, quantum_channel_wavelength, classical_rate, 
distance, collection_probability, quantum_channel_index, classical_channel_index):
    # double classical_travel, quantum_travel, transmissivity;
    # double decision;
    # double detection_time;

    noise_photons = []
    
    classical_speed = c/classical_channel_index
    quantum_speed = c/quantum_channel_index
    classical_attenuation = classical_channel_attenuation

    mean_num_photons = 2.2*classical_power * raman_coefficient* narrow_band_filter_bandwidth *   \
                        4 * np.exp(-classical_attenuation * distance/2) * np.sinh(classical_attenuation*pulse_width/2) * np.sinh(classical_attenuation*(distance-pulse_width)/2) *   \
                        quantum_channel_wavelength * quantum_channel_index /   \
                        (h*c*(c/1000)*classical_attenuation*classical_attenuation)
    

    # print("num_photons_added:", mean_num_photons)

    num_photons_added = np.random.poisson(mean_num_photons)

    for i in range(num_photons_added):

        classical_travel = -1/(classical_attenuation) * np.log(np.random.rand() * (np.exp(-distance*classical_attenuation) - 1) + 1)
        quantum_travel = direction*(distance-2*classical_travel)+classical_travel

        transmissivity = np.exp(-quantum_channel_attenuation * quantum_travel)

        decision = 0**(np.floor(np.random.rand()/(transmissivity*collection_probability)))
        detection_time = (symbol_number/(classical_rate/(2*1e12)) + (classical_travel*1000 / classical_speed + quantum_travel*1000 / quantum_speed) * 1e12);

        if decision:
            noise_photons.append(detection_time)

    return noise_photons



def transmit_classical_message(params, num_bits):
    """ adds a photon train of noisy photons scattered from the classical band into the quantum band."""
    directions = np.round(np.random.rand(2*num_bits))
    bits = np.round(np.random.rand(2*num_bits))

    kernel_file = open("classical_communication_kernel.cu", "r")
    cuda_kernel = r"{}".format(kernel_file.read())

    Raman_Kernel_call = cp.RawKernel( cuda_kernel, "Raman_Kernel" )

    max_raman_photons_per_pulse = 5 # Find this with some upper bound on number of photons scattered because 
                                    # by a single pulse

    # self.params["classical_powers"] = cp.array(self.params["classical_powers"], dtype = cp.float64)
    bits = cp.array(bits, dtype = cp.bool_)
    directions = cp.array(directions, dtype = cp.bool_)

    # print("directions:", directions)
    limit = int(len(bits)/2)

    # print("len of bits:", len(bits))

    noise_photons = cp.zeros((limit, max_raman_photons_per_pulse), dtype = cp.int64) # We don't know how many photons could be generated per pulse. Taking 5 per bit (on average) for now arbitrarily

    h = 6.62607015 * 10**(-34)
    c = 3. * 10**8
    # gpu_raman = cp.array(self.params["RAMAN_COEFFICIENTS"], dtype = cp.float64)
    # print("PYTHON: raman coeffs 2", self.params["RAMAN_COEFFICIENTS"], gpu_raman, "raman1:", self.params["RAMAN_COEFFICIENTS"][0])

    # test = [0.1e-7, 0.2e-8, 0.3e-9]
    # test_gpu = cp.array(test)
    # print("PYTHON: QUANTUM_WAVELENGTH = ", self.params["QUANTUM_WAVELENGTH"])

    # print("type of classical powers:", type(self.params["classical_powers"]))
    params = (bits, directions, noise_photons, limit, 
                cp.array(params["CLASSICAL_POWERS"], dtype = cp.float64),
                cp.array(params["RAMAN_COEFFICIENTS"], dtype = cp.float64),
                params["NBF_BANDWIDTH"],
                params["QUNATUM_ATTENUATION"] * np.log(10)/10,
                c/((params["CLASSICAL_RATE"]/2))/1e3,
                cp.array(params["CLASSICAL_ATTENUATION"], dtype = cp.float64) * np.log(10)/10,
                params["DIST_ANL_ERC"] * 1000 / c,
                h, c, 
                params["QUANTUM_WAVELENGTH"]*1e-9,
                params["CLASSICAL_RATE"],
                params["DIST_ANL_ERC"],
                params["BSM_DET1_EFFICIENCY"],
                params["QUANTUM_INDEX"],
                cp.array(params["CLASSICAL_INDEX"], dtype = cp.float64),
                max_raman_photons_per_pulse)

    # print("params: ", params[3:])
    # for i in params[3:]:
    #     print(i)
    
    # print("transmitting classical message")

    Raman_Kernel_call((limit//512+1,), (512,), params)

    cp.cuda.runtime.deviceSynchronize()




if __name__ == "__main__":
    h = 6.62607015 * 10**(-34)
    c = 3. * 10**8
    power = 10**( params["AVG_POWER"] /10)/1000
    OMA = 0.4*power 
    params["CLASSICAL_POWERS"] = np.array([[power-OMA/2, power-OMA/6, power+OMA/6, power+OMA/2], [power-OMA/2, power-OMA/6, power+OMA/6, power+OMA/2]])

    gpu_data_file = open("performance/gpu_data.csv", "a+")
    gpu_data_file.writelines(["Num bits",",", "Time", "\n"])

    cpu_data_file = open("performance/cpu_data.csv", "a+")
    cpu_data_file.writelines(["Num bits",",", "Time", "\n"])


    transmit_classical_message(params, 100)

    for i in np.logspace(2, 7, 11):
        num_bits = int(i)
        

        start_time = time.time()
        for symbol_number in range(num_bits):
            Raman_Kernel(symbol_number, np.round(np.random.rand()), np.random.choice(params["CLASSICAL_POWERS"][0]), 
                         params["RAMAN_COEFFICIENTS"][0], 
                         params["NBF_BANDWIDTH"],
                         params["QUNATUM_ATTENUATION"],
                         c/((params["CLASSICAL_RATE"]/2))/1e3,
                         params["CLASSICAL_ATTENUATION"][0],
                         h, c,
                         params["QUANTUM_WAVELENGTH"] * 10**-9,
                         params["CLASSICAL_RATE"],
                         params["DIST_ANL_ERC"],
                         params["BSM_DET1_EFFICIENCY"],
                         params["QUANTUM_INDEX"],
                         params["CLASSICAL_INDEX"][0]
                         )
        consumed_time = time.time()-start_time
        print(consumed_time)
        cpu_data_file.writelines([str(num_bits),",", str(consumed_time), "\n"])


        start_time = time.time()
        transmit_classical_message(params, num_bits)
        consumed_time = time.time()-start_time
        print(consumed_time)
        gpu_data_file.writelines([str(num_bits),",", str(consumed_time), "\n"])

        cpu_data_file.flush()
        gpu_data_file.flush()


        

        