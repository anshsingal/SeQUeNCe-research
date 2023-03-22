"""Definitions of node types.

This module provides definitions for various types of quantum network nodes.
All node types inherit from the base Node type, which inherits from Entity.
Node types can be used to collect all the necessary hardware and software for a network usage scenario.
"""

from math import inf
from time import monotonic_ns
from typing import TYPE_CHECKING, Any, List
import numpy as np

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline
    from ..message import Message
    from ..protocol import StackProtocol
    from ..resource_management.memory_manager import MemoryInfo
    from ..network_management.reservation import Reservation
    from ..components.optical_channel import QuantumChannel, ClassicalChannel
    from ..components.memory import Memory
    from ..app.random_request import RandomRequestApp

from ..kernel.entity import Entity
from ..components.memory import MemoryArray
from ..components.bsm import SingleAtomBSM
from ..components.light_source import LightSource
from ..components.detector import QSDetectorPolarization, QSDetectorTimeBin
from ..qkd.BB84 import BB84
from ..qkd.cascade import Cascade
from ..resource_management.resource_manager import ResourceManager
from ..network_management.network_manager import NewNetworkManager
from ..utils.encoding import *


class Node(Entity):
    """Base node type.
    
    Provides default interfaces for network.

    Attributes:
        name (str): label for node instance.
        timeline (Timeline): timeline for simulation.
        cchannels (Dict[str, ClassicalChannel]): mapping of destination node names to classical channel instances.
        qchannels (Dict[str, ClassicalChannel]): mapping of destination node names to quantum channel instances.
        protocols (List[Protocol]): list of attached protocols.
        generator (np.random.Generator): random number generator used by node.
    """

    def __init__(self, name: str, timeline: "Timeline", seed=None):
        """Constructor for node.

        name (str): name of node instance.
        timeline (Timeline): timeline for simulation.
        seed (int): seed for random number generator, default None
        """

        Entity.__init__(self, name, timeline)
        self.owner = self
        self.cchannels = {}  # mapping of destination node names to classical channels
        self.qchannels = {}  # mapping of destination node names to quantum channels
        self.protocols = []
        self.generator = np.random.default_rng(seed)

    def init(self) -> None:
        pass

    def set_seed(self, seed: int) -> None:
        self.generator = np.random.default_rng(seed)

    def get_generator(self):
        return self.generator

    def assign_cchannel(self, cchannel: "ClassicalChannel",
                        another: str) -> None:
        """Method to assign a classical channel to the node.

        This method is usually called by the `ClassicalChannel.add_ends` method and not called individually.

        Args:
            cchannel (ClassicalChannel): channel to add.
            another (str): name of node at other end of channel.
        """

        self.cchannels[another] = cchannel

    def assign_qchannel(self, qchannel: "QuantumChannel", another: str) -> None:
        """Method to assign a quantum channel to the node.

        This method is usually called by the `QuantumChannel.add_ends` method and not called individually.

        Args:
            qchannel (QuantumChannel): channel to add.
            another (str): name of node at other end of channel.
        """

        self.qchannels[another] = qchannel

    def send_message(self, dst: str, msg: "Message", priority=inf) -> None:
        """Method to send classical message.

        Args:
            dst (str): name of destination node for message.
            msg (Message): message to transmit.
            priority (int): priority for transmitted message (default inf).
        """

        if priority == inf:
            priority = self.timeline.schedule_counter
        self.cchannels[dst].transmit(msg, self, priority)

    def receive_message(self, src: str, msg: "Message") -> None:
        """Method to receive message from classical channel.

        Searches through attached protocols for those matching message, then invokes `received_message` method of protocol(s).

        Args:
            src (str): name of node sending the message.
            msg (Message): message transmitted from node.
        """

        # signal to protocol that we've received a message
        if msg.receiver is not None:
            for protocol in self.protocols:
                if protocol.name == msg.receiver and protocol.received_message(src, msg):
                    return
        else:
            matching = [p for p in self.protocols if type(p) == msg.protocol_type]
            for p in matching:
                p.received_message(src, msg)

    def schedule_qubit(self, dst: str, min_time: int) -> int:
        """Interface for quantum channel `schedule_transmit` method."""

        return self.qchannels[dst].schedule_transmit(min_time)

    def send_qubit(self, dst: str, qubit) -> None:
        """Interface for quantum channel `transmit` method."""

        print("sending qubits")

        self.qchannels[dst].transmit(qubit, self)

    def receive_qubit(self, src: str, qubit) -> None:
        """Method to receive qubits from quantum channel (does nothing for this class)."""

        pass


class BSMNode(Node):
    """Bell state measurement node.

    This node provides bell state measurement and the EntanglementGenerationB protocol for entanglement generation.

    Attributes:
        name (str): label for node instance.
        timeline (Timeline): timeline for simulation.
        bsm (SingleAtomBSM): BSM instance object.
        eg (EntanglementGenerationB): entanglement generation protocol instance.
    """

    def __init__(self, name: str, timeline: "Timeline",
                 other_nodes: List[str]) -> None:
        """Constructor for BSM node.

        Args:
            name (str): name of node.
            timeline (Timeline): simulation timeline.
            other_nodes (str): 2-member list of node names for adjacent quantum routers.
        """

        from ..entanglement_management.generation import \
            EntanglementGenerationB
        Node.__init__(self, name, timeline)
        self.bsm = SingleAtomBSM("%s_bsm" % name, timeline)
        self.bsm.owner = self
        self.eg = EntanglementGenerationB(self, "{}_eg".format(name),
                                          other_nodes)
        self.bsm.attach(self.eg)

    def receive_message(self, src: str, msg: "Message") -> None:
        # signal to protocol that we've received a message
        for protocol in self.protocols:
            if type(protocol) == msg.owner_type:
                if protocol.received_message(src, msg):
                    return

        # if we reach here, we didn't successfully receive the message in any protocol
        print(src, msg)
        raise Exception("Unkown protocol")

    def receive_qubit(self, src: str, qubit):
        """Method to receive qubit from quantum channel.

        Invokes get method of internal bsm with `qubit` as argument.

        Args:
            src (str): name of node where qubit was sent from.
            qubit (any): transmitted qubit.
        """

        self.bsm.get(qubit)

    def eg_add_others(self, other):
        """Method to addd other protocols to entanglement generation protocol.

        Args:
            other (EntanglementProtocol): other entanglement protocol instance.
        """

        self.eg.others.append(other.name)


class QuantumRouter(Node):
    """Node for entanglement distribution networks.

    This node type comes pre-equipped with memory hardware, along with the default SeQUeNCe modules (sans application).

    Attributes:
        name (str): label for node instance.
        timeline (Timeline): timeline for simulation.
        memory_array (MemoryArray): internal memory array object.
        resource_manager (ResourceManager): resource management module.
        network_manager (NetworkManager): network management module.
        map_to_middle_node (Dict[str, str]): mapping of router names to intermediate bsm node names.
        app (any): application in use on node.
    """

    def __init__(self, name, tl, memo_size=50):
        """Constructor for quantum router class.

        Args:
            name (str): label for node.
            tl (Timeline): timeline for simulation.
            memo_size (int): number of memories to add in the array (default 50).
        """

        Node.__init__(self, name, tl)
        self.memory_array = MemoryArray(name + ".MemoryArray", tl, num_memories=memo_size)
        self.memory_array.owner = self
        self.resource_manager = ResourceManager(self)
        self.network_manager = NewNetworkManager(self)
        self.map_to_middle_node = {}
        self.app = None

    def receive_message(self, src: str, msg: "Message") -> None:
        if msg.receiver == "resource_manager":
            self.resource_manager.received_message(src, msg)
        elif msg.receiver == "network_manager":
            self.network_manager.received_message(src, msg)
        else:
            if msg.receiver is None:
                matching = [p for p in self.protocols if type(p) == msg.protocol_type]
                for p in matching:
                    p.received_message(src, msg)
            else:
                for protocol in self.protocols:
                    if protocol.name == msg.receiver:
                        protocol.received_message(src, msg)
                        break

    def init(self):
        """Method to initialize quantum router node.

        Inherit parent function
        """

        super().init()

    def add_bsm_node(self, bsm_name: str, router_name: str):
        """Method to record connected BSM nodes

        Args:
            bsm_name (str): the BSM node between nodes self and router_name
            router_name (str): the name of another router connected with the BSM node
        """
        self.map_to_middle_node[router_name] = bsm_name

    def memory_expire(self, memory: "Memory") -> None:
        """Method to receive expired memories.

        Args:
            memory (Memory): memory that has expired.
        """

        self.resource_manager.memory_expire(memory)

    def set_app(self, app: "RandomRequestApp"):
        """Method to add an application to the node."""

        self.app = app

    def reserve_net_resource(self, responder: str, start_time: int, end_time: int, memory_size: int,
                             target_fidelity: float) -> None:
        """Method to request a reservation.

        Args:
            responder (str): name of the node with which entanglement is requested.
            start_time (int): desired simulation start time of entanglement.
            end_time (int): desired simulation end time of entanglement.
            memory_size (int): number of memories requested.
            target_fidelity (float): desired fidelity of entanglement.
        """

        self.network_manager.request(responder, start_time, end_time, memory_size, target_fidelity)

    def get_idle_memory(self, info: "MemoryInfo") -> None:
        """Method for application to receive available memories."""

        if self.app:
            self.app.get_memory(info)

    def get_reserve_res(self, reservation: "Reservation", res: bool) -> None:
        """Method for application to receive reservations results."""

        if self.app:
            self.app.get_reserve_res(reservation, res)

    def get_other_reservation(self, reservation: "Reservation"):
        """Method for application to get another reservation."""

        if self.app:
            self.app.get_other_reservation(reservation)


class QKDNode(Node):
    """Node for quantum key distribution.

    QKDNodes include a protocol stack to create keys.
    The protocol stack follows the "BBN QKD Protocol Suite" introduced in the DARPA quantum network
    (https://arxiv.org/pdf/quant-ph/0412029.pdf page 24).
    The protocol stack is:

    4. Authentication <= No implementation
    3. Privacy Amplification  <= No implementation
    2. Entropy Estimation <= No implementation
    1. Error Correction <= implemented by cascade
    0. Sifting <= implemented by BB84

    Attributes:
        name (str): label for node instance.
        timeline (Timeline): timeline for simulation.
        encoding (Dict[str, Any]): encoding type for qkd qubits (from encoding module).
        lightsource (LightSource): laser light source to generate keys.
        qsdetector (QSDetector): quantum state detector for qubit measurement.
        protocol_stack (List[StackProtocol]): protocols for qkdd process.
    """

    def __init__(self, name: str, timeline: "Timeline", encoding=polarization, stack_size=5):
        """Constructor for the qkd node class.

        Args:
            name (str): label for the node instance.
            timeline (Timeline): simulation timeline.
            encoding (Dict[str, Any]): encoding scheme for qubits (from encoding module) (default polarization).
            stack_size (int): number of qkd protocols to include in the protocol stack (default 5).
        """

        super().__init__(name, timeline)
        self.encoding = encoding
        self.lightsource = LightSource(name + ".lightsource", timeline, encoding_type=encoding)
        self.lightsource.owner = self

        if encoding["name"] == "polarization":
            self.qsdetector = QSDetectorPolarization(name + ".qsdetector", timeline)
        elif encoding["name"] == "time_bin":
            self.qsdetector = QSDetectorTimeBin(name + ".qsdetector", timeline)
        else:
            raise Exception("invalid encoding {} given for QKD node {}".format(encoding["name"], name))
        self.qsdetector.owner = self

        self.protocol_stack = [None] * 5

        if stack_size > 0:
            # Create BB84 protocol
            self.protocol_stack[0] = BB84(self, name + ".BB84")
            self.protocols.append(self.protocol_stack[0])

        if stack_size > 1:
            # Create cascade protocol
            self.protocol_stack[1] = Cascade(self, name + ".cascade")
            self.protocols.append(self.protocol_stack[1])
            self.protocol_stack[0].upper_protocols.append(self.protocol_stack[1])
            self.protocol_stack[1].lower_protocols.append(self.protocol_stack[0])

    def init(self) -> None:
        super().init()
        assert self.protocol_stack[0].role != -1

    def set_protocol_layer(self, layer: int, protocol: "StackProtocol") -> None:
        """Method to set a layer of the protocol stack.

        Args:
            layer (int): layer to change.
            protocol (StackProtocol): protocol to insert.
        """

        if layer < 0 or layer > 5:
            raise ValueError("layer must be between 0 and 5; given {}".format(layer))

        if self.protocol_stack[layer] is not None:
            self.protocols.remove(self.protocol_stack[layer])
        self.protocol_stack[layer] = protocol
        self.protocols.append(protocol)

        if layer > 0 and self.protocol_stack[layer - 1] is not None:
            self.protocol_stack[layer - 1].upper_protocols.append(protocol)
            protocol.lower_protocols.append(self.protocol_stack[layer - 1])

        if layer < 5 and self.protocol_stack[layer + 1] is not None:
            protocol.upper_protocols.append(self.protocol_stack[layer + 1])
            self.protocol_stack[layer + 1].lower_protocols.append(protocol)

    def update_lightsource_params(self, arg_name: str, value: Any) -> None:
        self.lightsource.__setattr__(arg_name, value)

    def update_detector_params(self, detector_id: int, arg_name: str, value: Any) -> None:
        self.qsdetector.update_detector_params(detector_id, arg_name, value)

    def get_bits(self, light_time, start_time, frequency):
        """Method for QKD protocols to get received qubits from the node.

        Uses the detection times from attached detectors to calculate which bits were received.
        Returns 0/1 for successfully transmitted bits and -1 for lost/ambiguous bits.

        Args:
            light_time (int): time for which qubits were transmitted.
            start_time (int): time at which qubits were first received.
            frequency (float): frequency of qubit transmission.

        Returns:
            List[int]: list of calculated bits.
        """

        # compute received bits based on encoding scheme
        encoding = self.encoding["name"]
        bits = [-1] * int(round(light_time * frequency))  # -1 used for invalid bits

        if encoding == "polarization":
            detection_times = self.qsdetector.get_photon_times()

            # determine indices from detection times and record bits
            for time in detection_times[0]:  # detection times for |0> detector
                index = round((time - start_time) * frequency * 1e-12)
                if 0 <= index < len(bits):
                    bits[index] = 0

            for time in detection_times[1]:  # detection times for |1> detector
                index = round((time - start_time) * frequency * 1e-12)
                if 0 <= index < len(bits):
                    if bits[index] == 0:
                        bits[index] = -1
                    else:
                        bits[index] = 1

        elif encoding == "time_bin":
            detection_times = self.qsdetector.get_photon_times()
            bin_separation = self.encoding["bin_separation"]
        
            # single detector (for early, late basis) times
            for time in detection_times[0]:
                index = int(round((time - start_time) * frequency * 1e-12))
                if 0 <= index < len(bits):
                    if abs(((index * 1e12 / frequency) + start_time) - time) < bin_separation / 2:
                        bits[index] = 0
                    elif abs(((index * 1e12 / frequency) + start_time) - (time - bin_separation)) < bin_separation / 2:
                        bits[index] = 1
        
            # interferometer detector 0 times
            for time in detection_times[1]:
                time -= bin_separation
                index = int(round((time - start_time) * frequency * 1e-12))
                # check if index is in range and is in correct time bin
                if 0 <= index < len(bits) and \
                        abs(((index * 1e12 / frequency) + start_time) - time) < bin_separation / 2:
                    if bits[index] == -1:
                        bits[index] = 0
                    else:
                        bits[index] = -1

            # interferometer detector 1 times
            for time in detection_times[2]:
                time -= bin_separation
                index = int(round((time - start_time) * frequency * 1e-12))
                # check if index is in range and is in correct time bin
                if 0 <= index < len(bits) and \
                        abs(((index * 1e12 / frequency) + start_time) - time) < bin_separation / 2:
                    if bits[index] == -1:
                        bits[index] = 1
                    else:
                        bits[index] = -1

        else:
            raise Exception("QKD node {} has illegal encoding type {}".format(self.name, encoding))

        return bits

    def set_bases(self, basis_list, start_time, frequency, component):
        """Method to set basis list for measurement component.

        Args:
            basis_list (List[int]): list of bases to measure in.
            start_time (int): time to start measurement.
            frequency (float): frequency with which to measure.
            component (Entity): measurement component to edit (normally a QSDetector).
        """

        encoding_type = component.encoding_type
        basis_start_time = start_time - 1e12 / (2 * frequency)

        if encoding_type["name"] == "polarization":
            splitter = component.splitter
            splitter.start_time = basis_start_time
            splitter.frequency = frequency

            splitter_basis_list = []
            for b in basis_list:
                splitter_basis_list.append(encoding_type["bases"][b])
            splitter.basis_list = splitter_basis_list

        elif encoding_type["name"] == "time_bin":
            switch = component.switch
            switch.start_time = basis_start_time
            switch.frequency = frequency
            switch.state_list = basis_list

        else:
            raise Exception("Invalid encoding type for node " + self.name)

    def receive_message(self, src: str, msg: "Message") -> None:
        # signal to protocol that we've received a message
        for protocol in self.protocols:
            if type(protocol) == msg.owner_type:
                protocol.received_message(src, msg)
                return

        # if we reach here, we didn't successfully receive the message in any protocol
        print(self.protocols)
        raise Exception("Message received for unknown protocol '{}' on node {}".format(msg.owner_type, self.name))

    def receive_qubit(self, src: str, qubit) -> None:
        self.qsdetector.get(qubit)
