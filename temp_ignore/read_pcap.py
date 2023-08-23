from scapy.all import *
from bitstring import BitArray
pcap = RawPcapNgReader("SeQUeNCe-0-0.pcap")
i = 0
for packet in pcap:
    
    bits = BitArray(packet[0])
    for i in bits:
        if i == 1:
            add_raman_photons()
        # print(int(i), end="")
    # print(str(bits))
    print("\n\none packet done")


def add_raman_photons():
    h = 6.62607015 * 10**(-34)
    c = 3 * 10**8
    clock_power = 0.0003
    raman_coefficient = 33e-10
    narrow_band_filter_bandwidth = 0.3
    attenuation = 0.1
    classical_channel_attenuation = 7
    quantum_channel_wavelength = 1536e-9

    window_size = 6.66e6
    pulse_width = 1e-5
    distance = 2

    raman_power = np.abs(clock_power * raman_coefficient * narrow_band_filter_bandwidth * (np.exp(-attenuation * pulse_width) - np.exp(-classical_channel_attenuation * pulse_width)) / (attenuation - classical_channel_attenuation))
    raman_energy = raman_power * window_size/1e12
    mean_num_photons = (raman_energy / (h * c / quantum_channel_wavelength))

    num_photons_added = sum(np.random.poisson(mean_num_photons, 50000))
    dAlpha = attenuation - classical_channel_attenuation
    positions = []
    # np.exp(distance * classical_channel_attenuation)
    for i in range(num_photons_added):
        positions.append((1/dAlpha) * np.log((np.exp(distance * dAlpha) - 1) * np.random.rand() + 1))

