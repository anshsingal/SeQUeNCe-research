import os
from scapy.all import PcapNgReader, raw
from bitstring import BitArray

os.system("cp NS3_script.cc /home/anshsingal/ns-allinone-3.38/ns-3.38/scratch/SeQUeNCe.cc && /home/anshsingal/ns-allinone-3.38/ns-3.38/ns3 run SeQUeNCe.cc")

# print("PCAP file name:", "pcap_files/SeQUeNCe-0-%s.pcap" % (self.sender_index))
pcap = PcapNgReader("pcap_files/SeQUeNCe-0-0.pcap")

for i in pcap:
    print("New packet")
    i.show()