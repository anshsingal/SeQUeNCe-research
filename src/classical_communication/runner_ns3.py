import os
import sys
from scapy.all import PcapNgReader, raw
from bitstring import BitArray

# Command line arguments required:
# Signature of run_NS3 function (except for verbose). 


def run_NS3(BER, NS3_file, tcp_file, udp_file, verbose = False):

    os.system(f'cp {NS3_file} {tcp_file} {udp_file} /home/asingal/ns-3-allinone/ns-3-dev/scratch/ && /home/asingal/ns-3-allinone/ns-3-dev/ns3 run "{NS3_file} {BER} {tcp_file} {udp_file}"')

    if verbose:
        # print("PCAP file name:", "pcap_files/SeQUeNCe-0-%s.pcap" % (self.sender_index))
        print("command line args:", sys.argv)
        pcap = PcapNgReader("pcap_files/temp-0-0.pcap")
        packet_counter = 0
        for i in pcap:
            packet_counter += 1
            # print("i.proto:", i.proto)
            # if i.proto == 'tcp':
            i.show()
        print("num_packets:", packet_counter)

if __name__ == "__main__":
    run_NS3(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], True)