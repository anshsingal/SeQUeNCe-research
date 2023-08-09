import os
from scapy.all import PcapNgReader, raw
from bitstring import BitArray

os.system("cp NS3_script.cc /home/asingal/ns-3-allinone/ns-3-dev/scratch/SeQUeNCe.cc && /home/asingal/ns-3-allinone/ns-3-dev/ns3 run SeQUeNCe.cc")

# print("PCAP file name:", "pcap_files/SeQUeNCe-0-%s.pcap" % (self.sender_index))
# pcap = PcapNgReader("pcap_files/SeQUeNCe-0-0.pcap")

# for i in pcap:
#     print("New packet")
#     i.show()