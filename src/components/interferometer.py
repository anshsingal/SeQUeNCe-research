"""Model for simulation of a Mach-Zehnder interferometer.

This module introduces a model for simulation of a Mach-Zehnder interferometer (MZI), used to measure time bin encoded qubits in the X-basis.
Interferometers are usually instantiated as part of a QSDetector object, defined in the `detector.py` module.
"""

from math import sqrt
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..kernel.timeline import Timeline
    from ..components.photon import Photon
    from ..components.detectors import Detector

from ..kernel.process import Process
from ..kernel.entity import Entity
from ..kernel.event import Event


class Interferometer(Entity):
    """Class modeling a Mach-Zehnder interferometer (MZI).

    Useful for measurement of time bin encoded photons in the X-basis.

    Attributes:
        name (str): label for beamsplitter instance
        timeline (Timeline): timeline for simulation
        path_difference (int): difference (in ps) of photon transit time in interferometer branches
        phase_error (float): phase error applied to measurement
        receivers (List[Entities]): entities to receive transmitted photons
    """

    def __init__(self, name: str, timeline: "Timeline", path_diff, phase_error=0):
        """Constructor for the interferometer class.

        Args:
            name (str): name of the interferometer instance.
            timeline (Timeline): simulation timeline.
            path_diff (int): sets path difference for interferometer instance.
            phase_error (float): phase error applied to measurement (default 0).
        """

        Entity.__init__(self, name, timeline)
        self.path_difference = path_diff  # time difference in ps
        self.phase_error = phase_error  # chance of measurement error in phase
        self.receivers = []

    def init(self) -> None:
        """See base class."""

        assert len(self.receivers) == 2

    def set_receiver(self, index: int, receiver: "Detector") -> None:
        """Sets the receivers attribute at the specified index."""

        if index > len(self.receivers):
            raise Exception("index is larger than the length of receivers")
        self.receivers.insert(index, receiver)

    def get(self, photon: "Photon") -> None:
        """Method to receive a photon for measurement.

        Arguments:
            photon (Photon): photon to measure (must have polarization encoding)

        Returns:
            None

        Side Effects:
            May call get method of one attached receiver from the receivers attribute.
        """
        detector_num = self.get_generator().choice([0, 1])
        quantum_state = photon.quantum_state
        time = 0
        random_num = self.get_generator().random()

        if quantum_state.state == (complex(1), complex(0)):  # Early
            if random_num <= 0.5:
                time = 0
            else:
                time = self.path_difference
        if quantum_state.state == (complex(0), complex(1)):  # Late
            if random_num <= 0.5:
                time = self.path_difference
            else:
                time = 2 * self.path_difference

        if self.get_generator().random() < self.phase_error:
            quantum_state.state = list(multiply([1, -1], quantum_state))

        if quantum_state.state == (complex(sqrt(1/2)), complex(sqrt(1/2))):  # Early + Late
            if random_num <= 0.25:
                time = 0
            elif random_num <= 0.5:
                time = 2 * self.path_difference
            elif detector_num == 0:
                time = self.path_difference
            else:
                return
        if quantum_state.state == (complex(sqrt(1/2)), complex(-sqrt(1/2))):  # Early - Late
            if random_num <= 0.25:
                time = 0
            elif random_num <= 0.5:
                time = 2 * self.path_difference
            elif detector_num == 1:
                time = self.path_difference
            else:
                return

        process = Process(self.receivers[detector_num], "get", [])
        event = Event(self.timeline.now() + time, process)
        self.timeline.schedule(event)
