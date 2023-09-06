/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/ipv4-address.h"
// #include <std/format>
#include "ns3/internet-trace-helper.h"

// Default Network Topology
//
//       10.1.1.0
// n0 -------------- n1
//    point-to-point
//

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FirstScriptExample");

int
main(int argc, char* argv[])
{
    CommandLine cmd(__FILE__);
    cmd.Parse(argc, argv);

    Time::SetResolution(Time::NS);
    LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
    LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);

    int num_nodes = 2;

    NodeContainer nodes;
    nodes.Create(num_nodes);

    PointToPointHelper pointToPoint;
    pointToPoint.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
    pointToPoint.SetChannelAttribute("Delay", StringValue("2ms"));

    NetDeviceContainer devices;
    devices = pointToPoint.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("192.168.1.0", "255.255.255.0");

    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    UdpEchoServerHelper echoServer(9);

    ApplicationContainer serverApps = echoServer.Install(nodes.Get(1));
    serverApps.Start(Seconds(1.0));
    serverApps.Stop(Seconds(10.0));

    UdpEchoClientHelper echoClient(interfaces.GetAddress(1), 9);
    echoClient.SetAttribute("MaxPackets", UintegerValue(1));
    echoClient.SetAttribute("Interval", TimeValue(Seconds(1.0)));
    echoClient.SetAttribute("PacketSize", UintegerValue(1024));

    ApplicationContainer clientApps = echoClient.Install(nodes.Get(0));
    clientApps.Start(Seconds(2.0));
    clientApps.Stop(Seconds(10.0));

    // Ptr<Ipv4> IPv4_Address;
    // uint8_t ip_address[6];
    // int  bytes_copied;
    // for (int i = 0; i < num_nodes; i++){
        // Ptr<NetDevice> p = container.Get (i)
        // IPv4_Address = devices.Get(i)-> GetObject<Ipv4>();
        // bytes_copied = p->GetAddress().CopyTo(ip_address);
        // printf("bytes copied:%d\n", bytes_copied);
        // // std::cout << "/home/anshsingal/SeQUeNCe-research/pcap_files/"+std::to_string(ip_address[5]);
        // p.EnablePcapIpv4("/home/anshsingal/SeQUeNCe-research/pcap_files/"+std::to_string(ip_address[5]), p, true, true);
        // i->method ();  // some NetDevice method
    // }
    // PcapHelperForIpv4 pcap_helper;
    // pcap_helper.EnablePcapIpv4("/home/anshsingal/work/", nodes);


    pointToPoint.EnablePcapAll("/home/anshsingal/SeQUeNCe-research/pcap_files/SeQUeNCe");
    // csma.EnablePcap("second", csmaDevices.Get(1), true);

    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
