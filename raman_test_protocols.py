from src.topology.node import Node
from src.protocol import Protocol
from src.message import Message

class RamanTestSender(Protocol):
    def __init__(self, own: Node, other_node: str):
        self.own = own
        own.protocols.append(self)
        self.other_node = other_node

    def start(self):
        # start clock and emissions
        pass

    def received_message(self, src: str, message: Message):
        pass


class RamanTestReceiver(Protocol):
    def __init__(self, own: Node, other_node: str):
        self.own = own
        own.protocols.append(self)
        self.other_node = other_node

    def start(self):
        # start clock and emissions
        pass

    def received_message(self, src: str, message: Message):
        pass