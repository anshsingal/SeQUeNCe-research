"""Code for Barrett-Kok entanglement Generation protocol

This module defines code to support entanglement generation between single-atom memories on distant nodes.
Also defined is the message type used by this implementation.
Entanglement generation is asymmetric:

* EntanglementGenerationA should be used on the QuantumRouter (with one node set as the primary) and should be started via the "start" method
* EntanglementGeneraitonB should be used on the BSMNode and does not need to be started
"""
from __future__ import annotations
from enum import Enum, auto
from math import sqrt
from typing import List, TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from ..components.memory import Memory
    from ..topology.node import Node, BSMNode
    from ..components.bsm import SingleAtomBSM

from .entanglement_protocol import EntanglementProtocol
from ..message import Message
from ..kernel.event import Event
from ..kernel.process import Process
from ..components.circuit import Circuit
from ..utils import log

class RamanTestProtocolEmitter(EntanglementProtocol):

    def __init__(self, own: Node, other: str):
        self.other = other
        self.own = own

    def start(self) -> None:
        "start clock and start emitting"

    def emit_event(self) -> None:
        """something similar required for emissions here too"""
        if self.ent_round == 1:
            self.memory.update_state(EntanglementGenerationA._plus_state)
        self.memory.excite(self.middle)


    def is_ready(self) -> bool:
        return self.remote_protocol_name is not None

class RamanTestProtocolReceiver(EntanglementProtocol):
    """Entanglement generation protocol for BSM node.

    The EntanglementGenerationB protocol should be instantiated on a BSM node.
    Instances will communicate with the A instance on neighboring quantum router nodes to generate entanglement.

    Attributes:
        own (BSMNode): node that protocol instance is attached to.
        name (str): label for protocol instance.
        others (List[str]): list of neighboring quantum router nodes
    """

    def __init__(self, own: BSMNode, name: str, others: List[str]):
        """Constructor for entanglement generation B protocol.

        Args:
            own (Node): attached node.
            name (str): name of protocol instance.
            others (List[str]): name of protocol instance on end nodes.
        """

        super().__init__(own, name)
        assert len(others) == 2
        self.others = others  # end nodes

    def bsm_update(self, bsm: SingleAtomBSM, info: Dict[str, Any]):
        """Method to receive detection events from BSM on node.

        Args:
            bsm (SingleAtomBSM): bsm object calling method.
            info (Dict[str, any]): information passed from bsm.
        """

        assert info['info_type'] == "BSM_res"

        res = info["res"]
        time = info["time"]
        resolution = bsm.resolution

        for i, node in enumerate(self.others):
            message = EntanglementGenerationMessage(GenerationMsgType.MEAS_RES,
                                                    None, detector=res,
                                                    time=time,
                                                    resolution=resolution)
            self.own.send_message(node, message)

    def received_message(self, src: str, msg: EntanglementGenerationMessage):
        raise Exception("EntanglementGenerationB protocol '{}' should not "
                        "receive message".format(self.name))

    def start(self) -> None:
        pass

    def set_others(self, protocol: str, node: str,
                   memories: List[str]) -> None:
        pass

    def is_ready(self) -> bool:
        return True

    def memory_expire(self, memory: "Memory") -> None:
        raise Exception("EntanglementGenerationB protocol '{}' should not have"
                        " memory_expire".format(self.name))
