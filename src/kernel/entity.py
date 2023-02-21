"""Definition of abstract Entity class.

This module defines the Entity class, inherited by all physical simulation elements (including hardware and photons).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict
from numpy.random import default_rng
from numpy.random._generator import Generator

if TYPE_CHECKING:
    from .timeline import Timeline


class Entity(ABC):
    """Abstract Entity class.
    Entity should use the provided pseudo random number generator (PRNG) to
    produce reproducible random numbers. As a result, simulations with the same
    seed could reproduce identical results. Function "get_generator" allows to
    get the PRNG.

    Attributes:
        name (str): name of the entity.
        timeline (Timeline): the simulation timeline for the entity.
        owner (Entity): another entity that owns or aggregates the current entity.
        _observer (List): a list of observers for the entity.
    """

    def __init__(self, name: str, timeline: "Timeline") -> None:
        """Constructor for entity class.

        Args:
            name (str): name of entity.
            timeline (Timeline): timeline for simulation.
        """
        self.name = "" if name is None else name
        self.timeline = timeline
        self.owner = None
        self._observers = []
        timeline.add_entity(self)

    @abstractmethod
    def init(self) -> None:
        """Method to initialize entity (abstract).

        Entity `init` methods are invoked for all timeline entities when the timeline is initialized.
        This method can be used to perform any necessary functions before simulation.
        """

        pass

    def attach(self, observer: Any) -> None:
        """Method to add an ovserver (to receive hardware updates)."""

        if not observer in self._observers:
            self._observers.append(observer)

    def detach(self, observer: Any) -> None:
        """Method to remove an observer."""

        self._observers.remove(observer)

    def notify(self, info: Dict[str, Any]) -> None:
        """Method to notify all attached observers of an update."""

        for observer in self._observers:
            observer.update(self, info)

    def remove_from_timeline(self) -> None:
        """Method to remove entity from attached timeline.

        This is to allow unused entities to be garbage collected.
        """
        self.timeline.remove_entity_by_name(self.name)

    def get_generator(self) -> Generator:
        """Method to get random generator of parent node.

        If entity is not attached to a node, return default generator.
        """
        if hasattr(self.owner, "generator"):
            return self.owner.get_generator()
        else:
            return default_rng()

    def change_timeline(self, timeline: "Timeline"):
        self.remove_from_timeline()
        self.timeline = timeline
        self.timeline.add_entity(self)
