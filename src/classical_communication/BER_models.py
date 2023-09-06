import math 
import numpy as np
from scipy.stats import norm

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

def calculate_BER_4PAM_noPreamplifier(classical_channel_attenuation, distance, B, power_levels, avg_power = 4e-3, T = 298, R_L = 50, F_n = 5, RIN = -155, R = 0.4):
    delta_f = B/2 # Receiver Bandwidth
    k_b = 1.38e-23
    e = 1.6e-19
    RIN = 10**(RIN/10) # We receive the RIN in dB
    received_power = lambda power: power*np.exp(-classical_channel_attenuation*distance) 
    # print("received power:", 10*np.log10(received_power(avg_power*1e3)))
    current = lambda power: R * received_power(power)
    thermal_noise = 4*k_b*T*F_n*delta_f/R_L
    level_noise = lambda current: thermal_noise + 2*e*current*delta_f + RIN*(current**2)*delta_f
    # power_levels = [avg_power-OMA/2, avg_power-OMA/6, avg_power+OMA/6, avg_power+OMA/2]
    total_noise = sum([level_noise( current(power) ) for power in power_levels])
    rms_noise = np.sqrt(total_noise)
    # print("total_noise:", total_noise, "RMS noise:", rms_noise, "avg_current:", current(avg_power), "parameter:", current(avg_power)/(3*np.sqrt(2)*rms_noise))
    SER = (0.75)*(math.erfc(current(avg_power)/(3*np.sqrt(2)*rms_noise)))
    hamming_distance_natural = 4/3
    BER = hamming_distance_natural * SER/np.log2(4)
    return BER

# Example:
# calculate_BER_4PAM_noPreamplifier(0.55, 2, 1.25e9, OMA = 0.5, avg_power = 1.5e-3)


# !!!!!!!!!!!!!!!!!!!!!!! This has not been implemented yet.!!!!!!!!!!!!!!!!!!!!!!!
#  It can be implementd by finding the SER for every adjacent pair of current levels. This SER corresponds to the BER in the Optical networks textbook 
# there, the symbol==bit (OOK). Summing all of those up should give us the the net SER which can in turn, give us the BER.  
def calculate_BER_4PAM_Preamplifier(classical_channel_attenuation, distance, B, OMA, avg_power = 4e-3, T = 298, R_L = 50, F_n = 5, RIN = -155, R = 0.4):
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