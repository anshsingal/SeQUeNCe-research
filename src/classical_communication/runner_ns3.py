import os
import sys
from scapy.all import PcapNgReader, raw
from bitstring import BitArray

# You need to provide the program with command line arguments. The first argument is the name of the NS3 file and the 
# second file is what you are reading from. 

print("command line args:", sys.argv)

os.system(f"cp {sys.argv[1]} {sys.argv[2]} /home/asingal/ns-3-allinone/ns-3-dev/scratch/ && /home/asingal/ns-3-allinone/ns-3-dev/ns3 run {sys.argv[1]}")

# print("PCAP file name:", "pcap_files/SeQUeNCe-0-%s.pcap" % (self.sender_index))
pcap = PcapNgReader("pcap_files/temp-0-0.pcap")
# print(len(pcap))
packet_counter = 0
for i in pcap:
    packet_counter += 1
    # print("New packet")
    i.show()
    # print("Seq:", i.seq, "Ack:", i.ack)

print("num_packets:", packet_counter)