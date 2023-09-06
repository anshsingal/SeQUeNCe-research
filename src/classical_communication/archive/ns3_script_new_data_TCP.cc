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
static uint32_t currentTxBytes = 0;

// Perform series of 1040 byte writes (this is a multiple of 26 since
// we want to detect data splicing in the output stream)
/// Write size.
static const uint32_t writeSize = 1040;
/// Data to be written.
uint8_t data[writeSize];
// Making the file so that it can be accessed multiple times by the WriteUntilBufferFull
// function without calling the fopen method again (maintains the position of the file pointer 
// within the file, not bringing it back to the start of the file.)
FILE *file;

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
void StartFlow(Ptr<Socket> localSocket, Ipv4Address servAddress, uint16_t servPort);

/**
 * Write to the buffer, filling it.
 *
 * \param localSocket The socket.
 * \param txSpace The number of bytes to write.
 */
void WriteUntilBufferFull(Ptr<Socket> localSocket, uint32_t TxBufferSize);

/**
 * Congestion window tracker function.
 *
 * \param oldval Old value.
 * \param newval New value.
 */
static void
CwndTracer(uint32_t oldval, uint32_t newval)
{
    NS_LOG_INFO("Moving cwnd from " << oldval << " to " << newval);
}

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
    // NetDeviceContainer dev1 = p2p.Install(n1n2);

    Ptr<RateErrorModel> em = CreateObject<RateErrorModel> ();
    // em->SetAttribute ("ErrorRate", DoubleValue (0.00001));
    printf("bit error rate is: %f\n", atof(argv[1]));
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

    ///////////////////////////////////////////////////////////////////////////
    // Simulation 1
    //
    // Send 2000000 bytes over a connection to server port 50000 at time 0
    // Should observe SYN exchange, a lot of data segments and ACKS, and FIN
    // exchange.  FIN exchange isn't quite compliant with TCP spec (see release
    // notes for more info)
    //
    ///////////////////////////////////////////////////////////////////////////

    uint16_t servPort = 50000;

    // Create a packet sink to receive these packets on n2...
    PacketSinkHelper sink("ns3::TcpSocketFactory",
                          InetSocketAddress(Ipv4Address::GetAny(), servPort));

    ApplicationContainer apps = sink.Install(n0n1.Get(1));
    apps.Start(Seconds(0.0));
    apps.Stop(Seconds(3.0));

    // Create a source to send packets from n0.  Instead of a full Application
    // and the helper APIs you might see in other example files, this example
    // will use sockets directly and register some socket callbacks as a sending
    // "Application".

    // Create and bind the socket...
    Ptr<Socket> localSocket = Socket::CreateSocket(n0n1.Get(0), TcpSocketFactory::GetTypeId());
    localSocket->Bind();

    // Trace changes to the congestion window
    Config::ConnectWithoutContext("/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow",
                                  MakeCallback(&CwndTracer));

    // ...and schedule the sending "Application"; This is similar to what an
    // ns3::Application subclass would do internally.
    Simulator::ScheduleNow(&StartFlow, localSocket, ipInterfs.GetAddress(1), servPort);

    // One can toggle the comment for the following line on or off to see the
    // effects of finite send buffer modelling.  One can also change the size of
    // said buffer.

    // localSocket->SetAttribute("SndBufSize", UintegerValue(4096));

    // Ask for ASCII and pcap traces of network traffic
    // AsciiTraceHelper ascii;
    // p2p.EnablePAll(ascii.CreateFileStream("tcp-large-transfer.tr"));
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
void StartFlow(Ptr<Socket> localSocket, Ipv4Address servAddress, uint16_t servPort)
{
    NS_LOG_LOGIC("Starting flow at time " << Simulator::Now().GetSeconds());
    std::cout<<"Starting flow at time " << Simulator::Now().GetSeconds()<<"\n";
    localSocket->Connect(InetSocketAddress(servAddress, servPort)); // connect
    TxBufferSize = localSocket->GetTxAvailable();
    
    printf("TxBufferSize is: %d\n", TxBufferSize);

    file = fopen("scratch/file.jpg", "r+"); 
    if(file == NULL){printf("could not open file\n");}

    // tell the tcp implementation to call WriteUntilBufferFull again
    // if we blocked and new tx buffer space becomes available
    localSocket->SetSendCallback(MakeCallback(&WriteUntilBufferFull));
    // printf("starting write process\n");
    WriteUntilBufferFull(localSocket, TxBufferSize);
}


// // Populating the packets with data. 
// void WriteUntilBufferFull(Ptr<Socket> localSocket, uint32_t txSpace)
// {
//     // printf("actually writing\n");
//     // currentTxBytes is a global variable, telling you how many bits have been sent already. 
//     // Make sure you dont send more data than the maximum you set in the main function.
//     // printf("socket dimensions are: %d\n", localSocket->GetTxAvailable());
//     printf("socket send buffer is: %d\n", localSocket->GetTxAvailable());
//     while (currentTxBytes < totalTxBytes && localSocket->GetTxAvailable() > 0) {
//         // Number of bits left to be sent
//         uint32_t left = totalTxBytes - currentTxBytes;

//         // Write size is the size of the buffer of 1040 charecters (writesize = 1040) from which 
//         // we take the data to write. 
//         uint32_t dataOffset = currentTxBytes % writeSize;
//         // printf("offset : %d\n", dataOffset);
        
//         // In regular operation, dataOffset = 0. This is because you generally write the entire writeSize
//         // worth of data  
//         uint32_t toWrite = writeSize - dataOffset;

//         // the number of charecters you write should not exceed the total 
//         // remaining number of bytes and also, you  eed to check if the transmit buffer
//         // can accomodate your request. So, you check that using GetTxAvailable.  
//         toWrite = std::min(toWrite, left);
//         toWrite = std::min(toWrite, localSocket->GetTxAvailable());

//         // This call does not create a packet and send it to the receiver. It just sends some data 
//         // to the socket to be sent in discrete packets. Creation of packets doesn't happen here. 
//         int amountSent = localSocket->Send(&data[dataOffset], toWrite, 0);
//         // printf("Amount sent : %d\n", amountSent);
//         if (amountSent < 0)
//         {
//             // we will be called again when new tx space becomes available.
//             return;
//         }
//         currentTxBytes += amountSent;
//     }
//     // If you have sent >= bits than the total bits that we are sending, close the socket. 
//     if (currentTxBytes >= totalTxBytes){
//         localSocket->Close();
//     }
// }

void WriteUntilBufferFull(Ptr<Socket> localSocket, uint32_t txSpacce) {

    // Three global objects used: TxBufferSize, currentTxBytes and file pointer

    // printf("GetTxAvailable:, %d\n", localSocket->GetTxAvailable());
    
    ///////////////////////REMOVE THIS/////////////////////
    // TxBufferSize = 7;
    
    uint8_t buffer[TxBufferSize];

    int num_bytes_available = 0; 
    int amountSent = 0;  
    int toWrite = 0;

    // Go to wherever we had sent the data until. Read the file from there. We should not need this in general
    // since our file pointer is global and the 
    // printf("TxAvailable before is: %d\n", localSocket->GetTxAvailable()); 

    fseek(file, currentTxBytes, SEEK_SET);
    num_bytes_available = fread(buffer, 1L, TxBufferSize, file);
    // printf("num_bytes_available: %d\n", num_bytes_available);
    // for(int i = 0; i<num_bytes_available; i++){
    //     printf("%d", buffer[i]);
    // }
    // printf("\n");
    // printf("this works\n");
    // printf("TxAvailable after is: %d\n", localSocket->GetTxAvailable());
    // int cond = 1;
    // printf("saw thr file\n");
    // printf("TxAvailable is: %d\n", localSocket->GetTxAvailable());

    while (currentTxBytes < totalTxBytes && localSocket->GetTxAvailable() > 0) {
        // Number of bits left to be sent
        toWrite = localSocket->GetTxAvailable();   
        //////////////////////REMOVE THIS//////////////////////////////////
        // toWrite = 5;
        // printf("toWrite: %d\n", toWrite);
        if (num_bytes_available < toWrite){ // There is more space in the socket TxBuffer than the file buffer. Send whatever we have and refill the file buffer
            // printf("not enough content to write.\n");
            amountSent = localSocket->Send(&buffer[TxBufferSize-num_bytes_available], num_bytes_available, 0);
            // printf("amountSent: %d\n", amountSent);
            fseek(file, currentTxBytes, SEEK_SET);
            num_bytes_available = fread(buffer, 1, TxBufferSize, file);
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

        currentTxBytes += amountSent;

        // printf("GetTxAvailable: %d\n", localSocket->GetTxAvailable());
        if (amountSent < 0) return;
    }
    // If you have sent >= bits than the total bits that we are sending, close the socket. 
    if (currentTxBytes >= totalTxBytes){
        localSocket->Close();
    }
}