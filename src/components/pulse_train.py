"""Model for single photon.

This module defines the Photon class for tracking individual photons.
Photons may be encoded directly with polarization or time bin schemes, or may herald the encoded state of single atom memories.
"""
from numpy.random._generator import Generator
from typing import Dict, Any, Optional
import numpy as np

from ..kernel.entity import Entity
from ..utils.encoding import polarization
from ..utils.quantum_state import QuantumState


class PulseTrain():
    """Class for a single photon.

    Attributes:
        name (str): label for photon instance.
        wavelength (float): wavelength of photon (in nm).
        location (Entity): current location of photon.
        encoding_type (Dict[str, Any]): encoding type of photon (as defined in encoding module).
        quantum_state (QuantumState): quantum state of photon.
        is_null (bool): defines whether photon is real or a "ghost" photon (not detectable but used in memory encoding).
    """

    def __init__(self, photon_counts, pulse_width, pulse_separation, wavelength, encoding_type=polarization):
        """Constructor for the photon class."""

        self.pulse_width = pulse_width
        self.pulse_separation = pulse_separation
        self.photon_counts = photon_counts
        self.wavelength = wavelength
        self.encoding_type = encoding_type
        self.pulse_count = len(photon_counts)

        self.time_offsets = np.arange(len(photon_counts)) * (self.pulse_separation + self.pulse_width)
        # self.time_offsets = self.time_offsets

        self.train_duration = self.time_offsets[-1]  + self.pulse_separation + self.pulse_width

        self.remove_vaccum()


    def __init__(self, time_offsets, train_duration, wavelength, encoding_type=polarization):
        """Constructor for the photon class."""

        self.wavelength = wavelength
        self.encoding_type = encoding_type
        # self.pulse_count = len(time_offsets)

        self.photon_counts = np.ones(len(time_offsets))

        self.train_duration = train_duration

        self.time_offsets = time_offsets


    def __init__(self):
        pass


    def copy(self):
        copied_pulse_train = PulseTrain()
        copied_pulse_train.pulse_width = self.pulse_width
        copied_pulse_train.pulse_separation = self.pulse_separation
        copied_pulse_train.photon_counts = self.photon_counts
        copied_pulse_train.wavelength = self.wavelength
        copied_pulse_train.encoding_type = self.encoding_type
        copied_pulse_train.pulse_count = self.pulse_count
        copied_pulse_train.time_offsets = self.time_offsets
        copied_pulse_train.train_duration = self.train_duration

        return copied_pulse_train



    
    def add_loss(self, loss_matrix):
        self.photon_counts = self.photon_counts - loss_matrix
        self.remove_vaccum() 

    def remove_vaccum(self):
        self.time_offsets = np.delete ( self.time_offsets, np.where(self.photon_counts == 0) )
        self.photon_counts = np.delete ( self.photon_counts, np.where(self.photon_counts == 0) )