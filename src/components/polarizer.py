from ..kernel.entity import Entity
from ..kernel.quantum_state import State 
import numpy as np


class Polarizer(Entity):
    
    def __init__(self, name, timeline, num_qubits = 1, angles = {0:0}):
        

        Entity.__init__(self, name, timeline)
        self.num_qubits = num_qubits
        self.rx = lambda theta:np.array([ [np.cos(theta/2), np.sin(theta/2)], [-np.sin(theta/2), np.cos(theta/2)] ])
        self.identity = np.eye(2)
        self.ket_0 = np.array([1,0])[np.newaxis].T
        self.rotate(angles)


    def init(self):
        """Implementation of Entity interface (see base class)."""

        pass

    def rotate(self, new_angles_dict):
        """
        Rotate the polarizers by angles specified in dictionary new_angles_dict. 
        The contents of the dict are: {qubit_index : polarizer angle}
        For the qubit indices not mapped to a polarizer angle, the identity operator
        would be applied to the qubit. 
        """


        self.projector = 1

        for i in range(self.num_qubits):
            if i in new_angles_dict.keys():
                reference_state = self.rx(new_angles_dict[i])   @   self.ket_0
                self.projector = np.kron(self.projector, reference_state @ np.conjugate(reference_state.T)) 
            else:
                self.projector = np.kron(self.projector, self.identity)
        # self.projector = tuple(map(tuple, self.projector)) 
        # print("projector:")
        # print(self.projector)
        # print("rx:")
        # for i in new_angles_dict.keys():
        #     print()
        #     print(self.rx(new_angles_dict[i]))
            
        # print(self.rx(new_angles_dict[1]))



    def get(self, photon, **kwargs):
        if not isinstance(photon, State):
            photon = photon.quantum_state
        # print("state:", type(self.projector @ photon.state), (self.projector @ photon.state).shape)
        if photon.density_matrix:
            photon.set_state(self.projector @ photon.state @ self.projector, density_matrix = True)
        else:
            photon.set_state(self.projector @ photon.state)
        # print("self.num_qubits:", self.num_qubits)
        return
        
        # quantum_state = self.timeline.quantum_manager.get(photon.quantum_state)
        # photon.set_state(self.projector @ quantum_state)
        # self._receivers[0].get(photon)
