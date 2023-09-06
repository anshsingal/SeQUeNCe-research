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
 *
 */

//
// Network topology
//
//           10Mb/s, 10ms       10Mb/s, 10ms
//       n0-----------------n1-----------------n2
//
//
// - Tracing of queues and packet receptions to file
//   "tcp-large-transfer.tr"
// - pcap traces also generated in the following files
//   "tcp-large-transfer-$n-$i.pcap" where n and i represent node and interface
// numbers respectively
//  Usage (e.g.): ./ns3 run tcp-large-transfer

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/error-model.h"

#include <fstream>
#include <iostream>
#include <string>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpLargeTransfer");

/// The number of bytes to send in this simulation.
static const uint32_t totalTxBytes = 2000000;
/// The actual number of sent bytes.
static uint32_t currentTxBytesTCP = 0, currentTxBytesTCP2 = 0, currentTxBytesUDP = 0;

// Perform series of 1040 byte writes (this is a multiple of 26 since
// we want to detect data splicing in the output stream)
/// Write size.
static const uint32_t writeSize = 1040;

/// Data to be written.
uint8_t data[writeSize];
// Making the file so that it can be accessed multiple times by the WriteUntilBufferFull
// function without calling the fopen method again (maintains the position of the file pointer 
// within the file, not bringing it back to the start of the file.)
FILE *TCPfile, *UDPfile;

// This is the socket Trsnamit buffer size. Making it global to know the maximum size of 
// transmit buffer every time we open it. 
int TxBufferSize;

// These are for starting the writing process, and handling the sending
// socket's notification upcalls (events).  These two together more or less
// implement a sending "Application", although not a proper ns3::Application
// subclass.

/**
 * Start a flow.
 *
 * \param localSocket The local (sending) socket.
 * \param servAddress The server address.
 * \param servPort The server port.
 */
void StartFlow(Ptr<Socket> localSocket, Ipv4Address servAddress, uint16_t servPort, bool isTCP);
void StartFlow2(Ptr<Socket> localSocket, Ipv4Address servAddress, uint16_t servPort, bool isTCP);

/**
 * Write to the buffer, filling it.
 *
 * \param localSocket The socket.
 * \param txSpace The number of bytes to write.
 */
void TCPWriteUntilBufferFull(Ptr<Socket> localSocket, uint32_t TxBufferSize);
void UDPWriteUntilBufferFull(Ptr<Socket> localSocket, uint32_t TxBufferSize);
void TCPWriteUntilBufferFull2(Ptr<Socket> localSocket, uint32_t txSpacce);

// /**
//  * Congestion window tracker function.
//  *
//  * \param oldval Old value.
//  * \param newval New value.
//  */
// static void
// CwndTracer(uint32_t oldval, uint32_t newval)
// {
//     NS_LOG_INFO("Moving cwnd from " << oldval << " to " << newval);
// }

int main(int argc, char* argv[]) {
    // Users may find it convenient to turn on explicit debugging
    // for selected modules; the below lines suggest how to do this
    //  LogComponentEnable("TcpL4Protocol", LOG_LEVEL_ALL);
    //  LogComponentEnable("TcpSocketImpl", LOG_LEVEL_ALL);
    //  LogComponentEnable("PacketSink", LOG_LEVEL_ALL);
    //  LogComponentEnable("TcpLargeTransfer", LOG_LEVEL_ALL);
    // printf("program has started\n");


    CommandLine cmd(__FILE__);
    cmd.Parse(argc, argv);

    // printf("command line arguments: %s, %s, %s \n", argv[0], argv[1], argv[2]);

    // printf("Launch power is: %d\n", atoi(argv[1]));

    // initialize the tx buffer.
    // for (uint32_t i = 0; i < writeSize; ++i)
    // {
    //     char m = toascii(97 + i % 26);
    //     data[i] = m;
    // }

    // Here, we will explicitly create three nodes.  The first container contains
    // nodes 0 and 1 from the diagram above, and the second one contains nodes
    // 1 and 2.  This reflects the channel connectivity, and will be used to
    // install the network interfaces and connect them with a channel.
    int num_nodes = 2;
    NodeContainer n0n1;
    n0n1.Create(num_nodes);

    // NodeContainer n1n2;
    // n1n2.Add(n0n1.Get(1));
    // n1n2.Create(1);

    // We create the channels first without any IP addressing information
    // First make and configure the helper, so that it will put the appropriate
    // attributes on the network interfaces and channels we are about to install.
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", DataRateValue(DataRate(10000000)));
    p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(10)));

    // And then install devices and channels connecting our topology.
    NetDeviceContainer dev0 = p2p.Install(n0n1);

    // Create Error model object and apply it to the receiver
    Ptr<RateErrorModel> em = CreateObject<RateErrorModel> ();
    em->SetRate(atof(argv[1]));
    em->SetUnit(RateErrorModel::ERROR_UNIT_BIT);

    dev0.Get (1)->SetAttribute ("ReceiveErrorModel", PointerValue (em));


    // Now add ip/tcp stack to all nodes.
    InternetStackHelper internet;
    internet.InstallAll();

    // Later, we add IP addresses.
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer ipInterfs = ipv4.Assign(dev0);
    // ipv4.SetBase("10.1.2.0", "255.255.255.0");
    // Ipv4InterfaceContainer ipInterfs = ipv4.Assign(dev1);

    // and setup ip routing tables to get total ip-level connectivity.
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();


    char tcp_directory[20] = "scratch/", udp_directory[20] = "scratch/";
    strcat(tcp_directory, argv[2]);
    strcat(udp_directory, argv[3]);
    TCPfile = fopen(tcp_directory, "r+");
    UDPfile = fopen(udp_directory, "r+");

    //
    // Setting up communications: UDP and TCP between the two nodes. 
    //

    // // Setting up UDP communication from node0 -> node1
    // uint16_t UDPservPort = 50000;
    
    // // Create UDP source socket on the source node (node0). 
    // Ptr<Socket> UDPlocalSocket = Socket::CreateSocket(n0n1.Get(0), UdpSocketFactory::GetTypeId());
    // UDPlocalSocket->Bind();

    // // Create a sink application (standardised) on the receiver node (node1). We use a helper to install the application for us. You give it the UDPservPort to know where to listen. 
    // PacketSinkHelper UDPsink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), UDPservPort)); 
    
    // ApplicationContainer UDPapps = UDPsink.Install(n0n1.Get(1));
    // UDPapps.Start(Seconds(0.0));
    // UDPapps.Stop(Seconds(3.0));

    // Setting up UDP communication from node0 -> node1
    uint16_t TCPservPort2 = 50000;
    
    // Create UDP source socket on the source node (node0). 
    Ptr<Socket> TCPlocalSocket2 = Socket::CreateSocket(n0n1.Get(0), TcpSocketFactory::GetTypeId());
    TCPlocalSocket2->Bind();

    // Create a sink application (standardised) on the receiver node (node1). We use a helper to install the application for us. You give it the UDPservPort to know where to listen. 
    PacketSinkHelper TCPsink2("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), TCPservPort2)); 
    
    ApplicationContainer TCPapps2 = TCPsink2.Install(n0n1.Get(1));
    TCPapps2.Start(Seconds(0.0));
    TCPapps2.Stop(Seconds(3.0));




    // Setting up TCP communication from node1 -> node0
    uint16_t TCPservPort = 60000;

    // Setup TCPSocket on the sender node (node1). 
    Ptr<Socket> TCPlocalSocket = Socket::CreateSocket(n0n1.Get(1), TcpSocketFactory::GetTypeId());
    TCPlocalSocket->Bind();

    // Create TCP sink application on receiver node (Node0)
    PacketSinkHelper TCPsink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), TCPservPort));

    ApplicationContainer TCPapps = TCPsink.Install(n0n1.Get(0));
    TCPapps.Start(Seconds(0.0));
    TCPapps.Stop(Seconds(3.0));


    // Schedule the TCP protocols. 
    Simulator::ScheduleNow(&StartFlow, TCPlocalSocket, ipInterfs.GetAddress(0), TCPservPort, true);
    // Schedule the UDP communicartion 
    Simulator::ScheduleNow(&StartFlow2, TCPlocalSocket2, ipInterfs.GetAddress(1), TCPservPort2, false);

    // // Setting up TCP communication from node1 -> node0
    // uint16_t UDPservPort2 = 60000;

    // // Setup TCPSocket on the sender node (node1). 
    // Ptr<Socket> UDPlocalSocket2 = Socket::CreateSocket(n0n1.Get(1), UdpSocketFactory::GetTypeId());
    // UDPlocalSocket2->Bind();

    // // Create TCP sink application on receiver node (Node0)
    // PacketSinkHelper UDPsink2("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), UDPservPort2));

    // ApplicationContainer UDPapps2 = UDPsink2.Install(n0n1.Get(0));
    // UDPapps2.Start(Seconds(0.0));
    // UDPapps2.Stop(Seconds(3.0));

    // // Schedule the TCP protocols. 
    // Simulator::ScheduleNow(&StartFlow, UDPlocalSocket2, ipInterfs.GetAddress(0), UDPservPort2, false);




    p2p.EnablePcapAll("/home/asingal/Visibility-SeQUeNCe-research/src/classical_communication/pcap_files/temp");

    // Finally, set up the simulator to run.  The 1000 second hard limit is a
    // failsafe in case some change above causes the simulation to never end
    Simulator::Stop(Seconds(1000));
    Simulator::Run();
    Simulator::Destroy();

    return 0;
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// begin implementation of sending "Application"
void StartFlow(Ptr<Socket> localSocket, Ipv4Address servAddress, uint16_t servPort, bool isTCP)
{
    // NS_LOG_LOGIC("Starting flow at time " << Simulator::Now().GetSeconds());
    // std::cout<<"Starting flow at time " << Simulator::Now().GetSeconds()<<"\n";
    localSocket->Connect(InetSocketAddress(servAddress, servPort)); // connect
    TxBufferSize = localSocket->GetTxAvailable();
    
    printf("TxBufferSize is: %d\n", TxBufferSize);

    if(TCPfile == NULL || UDPfile == NULL){printf("could not open file\n");}

    // tell the tcp implementation to call WriteUntilBufferFull again
    // if we blocked and new tx buffer space becomes available
    if (isTCP){
        localSocket->SetSendCallback(MakeCallback(&TCPWriteUntilBufferFull));
        TCPWriteUntilBufferFull(localSocket, TxBufferSize);
    }
    else{
        localSocket->SetSendCallback(MakeCallback(&UDPWriteUntilBufferFull));
        UDPWriteUntilBufferFull(localSocket, TxBufferSize);    
    }
}

void StartFlow2(Ptr<Socket> localSocket, Ipv4Address servAddress, uint16_t servPort, bool isTCP)
{
    // NS_LOG_LOGIC("Starting flow at time " << Simulator::Now().GetSeconds());
    // std::cout<<"Starting flow at time " << Simulator::Now().GetSeconds()<<"\n";
    localSocket->Connect(InetSocketAddress(servAddress, servPort)); // connect
    TxBufferSize = localSocket->GetTxAvailable();
    
    printf("TxBufferSize is: %d\n", TxBufferSize);

    if(TCPfile == NULL || UDPfile == NULL){printf("could not open file\n");}

    // tell the tcp implementation to call WriteUntilBufferFull again
    // if we blocked and new tx buffer space becomes available
    if (isTCP){
        localSocket->SetSendCallback(MakeCallback(&TCPWriteUntilBufferFull2));
        TCPWriteUntilBufferFull2(localSocket, TxBufferSize);
    }
    else{
        localSocket->SetSendCallback(MakeCallback(&UDPWriteUntilBufferFull));
        UDPWriteUntilBufferFull(localSocket, TxBufferSize);    
    }
}


void TCPWriteUntilBufferFull(Ptr<Socket> localSocket, uint32_t txSpacce) {

    // Three global objects used: TxBufferSize, currentTxBytesTCP and file pointer

    uint8_t buffer[TxBufferSize];

    int num_bytes_available = 0; 
    int amountSent = 0;  
    int toWrite = 0;

    fseek(TCPfile, currentTxBytesTCP, SEEK_SET);
    num_bytes_available = fread(buffer, 1L, TxBufferSize, TCPfile);

    while (currentTxBytesTCP < totalTxBytes && localSocket->GetTxAvailable() > 0) {
        // Number of bits left to be sent
        toWrite = localSocket->GetTxAvailable();   
        printf("toWrite is: %d\n", toWrite);

        if (num_bytes_available < toWrite){ // There is more space in the socket TxBuffer than the file buffer. Send whatever we have and refill the file buffer
            // printf("not enough content to write.\n");
            amountSent = localSocket->Send(&buffer[TxBufferSize-num_bytes_available], num_bytes_available, 0);
            printf("TCPWriteUntilBufferFull sent %d\n", amountSent);
            // printf("amountSent: %d\n", amountSent);
            fseek(TCPfile, currentTxBytesTCP, SEEK_SET);
            num_bytes_available = fread(buffer, 1, TxBufferSize, TCPfile);
            // printf("new bytes available: %d\n", num_bytes_available);
            // The file is finished.
            if (num_bytes_available == 0) return;
        }
        else{ // We have sufficient amount of data in file buffer. Go ahead with sending from it. 
            // printf("we have enough data\n");
            amountSent = localSocket->Send(&buffer[TxBufferSize-num_bytes_available], toWrite, 0);
            num_bytes_available -= toWrite;
            // printf("num_bytes_available: %d\n", num_bytes_available);
        }

        currentTxBytesTCP += amountSent;

        // printf("GetTxAvailable: %d\n", localSocket->GetTxAvailable());
        if (amountSent < 0) return;
    }
    // If you have sent >= bits than the total bits that we are sending, close the socket. 
    if (currentTxBytesTCP >= totalTxBytes){
        localSocket->Close();
    }
}




void TCPWriteUntilBufferFull2(Ptr<Socket> localSocket, uint32_t txSpacce) {

    // Three global objects used: TxBufferSize, currentTxBytesTCP and file pointer

    uint8_t buffer[TxBufferSize];

    int num_bytes_available = 0; 
    int amountSent = 0;  
    int toWrite = 0;

    fseek(TCPfile, currentTxBytesTCP2, SEEK_SET);
    num_bytes_available = fread(buffer, 1L, TxBufferSize, TCPfile);

    while (currentTxBytesTCP2 < totalTxBytes && localSocket->GetTxAvailable() > 0) {
        // Number of bits left to be sent
        toWrite = localSocket->GetTxAvailable();   
        printf("toWrite is: %d\n", toWrite);

        if (num_bytes_available < toWrite){ // There is more space in the socket TxBuffer than the file buffer. Send whatever we have and refill the file buffer
            // printf("not enough content to write.\n");
            amountSent = localSocket->Send(&buffer[TxBufferSize-num_bytes_available], num_bytes_available, 0);
            printf("TCPWriteUntilBufferFull sent %d\n", amountSent);
            // printf("amountSent: %d\n", amountSent);
            fseek(TCPfile, currentTxBytesTCP2, SEEK_SET);
            num_bytes_available = fread(buffer, 1, TxBufferSize, TCPfile);
            // printf("new bytes available: %d\n", num_bytes_available);
            // The file is finished.
            if (num_bytes_available == 0) return;
        }
        else{ // We have sufficient amount of data in file buffer. Go ahead with sending from it. 
            // printf("we have enough data\n");
            amountSent = localSocket->Send(&buffer[TxBufferSize-num_bytes_available], toWrite, 0);
            num_bytes_available -= toWrite;
            // printf("num_bytes_available: %d\n", num_bytes_available);
        }

        currentTxBytesTCP2 += amountSent;

        // printf("GetTxAvailable: %d\n", localSocket->GetTxAvailable());
        if (amountSent < 0) return;
    }
    // If you have sent >= bits than the total bits that we are sending, close the socket. 
    if (currentTxBytesTCP2 >= totalTxBytes){
        localSocket->Close();
    }
}





// This is not correct. It isn't wrong, but UDP sockets don't have variable buffer sizes but constant
// datagram sizes and hence, the same amount of data must transmitted every time.  
void UDPWriteUntilBufferFull(Ptr<Socket> localSocket, uint32_t txSpacce) {

    // Three global objects used: TxBufferSize, currentTxBytes and file pointer

    uint8_t buffer[TxBufferSize];

    int num_bytes_available = 0; 
    int amountSent = 0;  
    int toWrite = 0;

    fseek(UDPfile, currentTxBytesUDP, SEEK_SET);
    num_bytes_available = fread(buffer, 1L, TxBufferSize, UDPfile);

    while (currentTxBytesUDP < totalTxBytes && localSocket->GetTxAvailable() > 0) {
        // Number of bits left to be sent
        toWrite = localSocket->GetTxAvailable();   

        if (num_bytes_available < toWrite){ // There is more space in the socket TxBuffer than the file buffer. Send whatever we have and refill the file buffer
            // printf("not enough content to write.\n");
            amountSent = localSocket->Send(&buffer[TxBufferSize-num_bytes_available], num_bytes_available, 0);
            // printf("amountSent: %d\n", amountSent);
            fseek(UDPfile, currentTxBytesUDP, SEEK_SET);
            num_bytes_available = fread(buffer, 1, TxBufferSize, UDPfile);
            // printf("new bytes available: %d\n", num_bytes_available);
            // The file is finished.
            if (num_bytes_available == 0) return;
        }
        else{ // We have sufficient amount of data in file buffer. Go ahead with sending from it. 
            // printf("we have enough data\n");
            amountSent = localSocket->Send(&buffer[TxBufferSize-num_bytes_available], toWrite, 0);
            num_bytes_available -= toWrite;
            // printf("num_bytes_available: %d\n", num_bytes_available);
        }

        currentTxBytesUDP += amountSent;

        // printf("GetTxAvailable: %d\n", localSocket->GetTxAvailable());
        if (amountSent < 0) return;
    }
    // If you have sent >= bits than the total bits that we are sending, close the socket. 
    if (currentTxBytesUDP >= totalTxBytes){
        localSocket->Close();
    }
}