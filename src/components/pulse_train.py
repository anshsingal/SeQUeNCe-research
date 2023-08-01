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


class PulseWindow():
    def __init__(self, ID):
        self.source_train = []
        self.noise_train = []
        self.other_trains = []
        self.ID = ID

class PulseTrain():

    def __init__(self, *args):
        if len(args) == 4:
            self.init1(*args)
        elif len(args) == 2:
            self.init2(*args)
        else:
            pass

    def init1(self, photon_counts, pulse_width, pulse_separation, wavelength, encoding_type=polarization):
        """Constructor for the pulse train based on pulse seperation, width and photon counts. Meant for the light source to use"""

        self.pulse_width = pulse_width
        self.pulse_separation = pulse_separation
        self.photon_counts = photon_counts
        self.wavelength = wavelength
        self.encoding_type = encoding_type
        self.pulse_count = len(photon_counts)

        self.time_offsets = np.arange(len(photon_counts)) * (self.pulse_separation + self.pulse_width)
        self.train_duration = self.time_offsets[-1]  + self.pulse_separation + self.pulse_width
        self.remove_vaccum()


    def init2(self, time_offsets, train_duration, encoding_type=polarization):
        """Generalized constrcutor where the calling function provides the relative time offsets between the photons in the train"""

        # self.wavelength = wavelength
        self.encoding_type = encoding_type
        self.photon_counts = np.ones(len(time_offsets), dtype = "int")
        self.train_duration = train_duration
        self.time_offsets = np.array(time_offsets)



    def copy(self):
        """ Copies the pulse train's attributed to create an exact clone of the pulse train.
        
            Used by:
                Light source to create the idler train based on the signal train 
        
        """
        copied_pulse_train = PulseTrain()
        copied_pulse_train.pulse_width = self.pulse_width
        copied_pulse_train.pulse_separation = self.pulse_separation
        copied_pulse_train.photon_counts = self.photon_counts.copy()
        copied_pulse_train.wavelength = self.wavelength
        copied_pulse_train.encoding_type = self.encoding_type
        copied_pulse_train.pulse_count = self.pulse_count
        copied_pulse_train.time_offsets = self.time_offsets.copy()
        copied_pulse_train.train_duration = self.train_duration

        return copied_pulse_train


    
    def add_loss(self, loss_matrix):
        self.photon_counts = self.photon_counts - loss_matrix
        self.remove_vaccum() 

    def remove_vaccum(self):
        self.time_offsets = np.delete ( self.time_offsets, np.where(self.photon_counts == 0) )
        self.photon_counts = np.delete ( self.photon_counts, np.where(self.photon_counts == 0) )