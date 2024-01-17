"""Definition of the quantum state classes.

This module defines the classes used to track quantum states in SeQUeNCe.
These include 2 classes used by a quantum manager, and one used for individual photons:

1. The `KetState` class represents the ket vector formalism and is used by a quantum manager.
2. The `DensityState` class represents the density matrix formalism and is also used by a quantum manager.
3. The `FreeQuantumState` class uses the ket vector formalism, and is used by individual photons (not the quantum manager).
"""

from abc import ABC
from typing import Tuple, Dict, List

from numpy import pi, cos, sin, arange, log, log2
from numpy.random import Generator
import numpy as np

from .quantum_utils import *
import scipy.sparse as sp


def swap_bits(num, pos1, pos2):
    """Swaps bits in num at positions 1 and 2.

    Used by quantum_state.measure_multiple method.
    """

    bit1 = (num >> pos1) & 1
    bit2 = (num >> pos2) & 1
    x = bit1 ^ bit2
    x = (x << pos1) | (x << pos2)
    return num ^ x


class State(ABC):
    """Base class for storing quantum states (abstract).

    Attributes:
        state (any): internal representation of the state, may vary by state type.
        keys (List[int]): list of keys pointing to the state, for use with a quantum manager.
    """

    def __init__(self, **kwargs):
        # potential key word arguments for derived classes, e.g. truncation = d-1 for qudit

        super().__init__()

        self.state = None
        self.keys = []

    def deserialize(self, json_data) -> None:
        self.keys = json_data["keys"]
        self.state = []
        for i in range(0, len(json_data["state"]), 2):
            complex_val = complex(json_data["state"][i],
                                  json_data["state"][i + 1])
            self.state.append(complex_val)

    def serialize(self) -> Dict:
        res = {"keys": self.keys}
        state = []
        for cplx_n in self.state:
            if type(cplx_n) == float:
                state.append(cplx_n)
                state.append(0)
            elif isinstance(cplx_n, complex):
                state.append(cplx_n.real)
                state.append(cplx_n.imag)
            else:
                raise ValueError("Unknown type of state")

        res["state"] = state
        return res

    def __str__(self):
        return "\n".join(["Keys:", str(self.keys), "State:", str(self.state)])


class KetState(State):
    """Class to represent an individual quantum state as a ket vector.

    Attributes:
        state (np.array): state vector. Should be of length d ** len(keys), where d is dimension of elementary
            Hilbert space. Default is 2 for qubits.
        keys (List[int]): list of keys (subsystems) associated with this state.
        truncation (int): maximally allowed number of excited states for elementary subsystems.
                Default is 1 for qubit. dim = truncation + 1
    """

    def __init__(self, amplitudes: List[complex], keys: List[int], truncation: int = 1):
        """Constructor for ket state class.

        Args:
            amplitudes
            truncation (int): maximally allowed number of excited states for elementary subsystems.
                Default is 1 for qubit. dim = truncation + 1
        """
        super().__init__()
        self.truncation = truncation
        dim = self.truncation + 1  # dimension of element Hilbert space

        # check formatting
        assert all([abs(a) <= 1.01 for a in amplitudes]), "Illegal value with abs > 1 in ket vector"
        assert abs(sum([abs(a) ** 2 for a in amplitudes]) - 1) < 1e-5, "Squared amplitudes do not sum to 1"

        num_subsystems = log(len(amplitudes)) / log(dim)
        assert dim ** int(round(num_subsystems)) == len(amplitudes),\
            "Length of amplitudes should be d ** n, " \
            "where d is subsystem Hilbert space dimension and n is the number of subsystems. " \
            "Actual amplitude length: {}, dim: {}, num subsystems: {}".format(
                len(amplitudes), dim, num_subsystems
            )
        num_subsystems = int(round(num_subsystems))
        assert num_subsystems == len(keys),\
            "Length of amplitudes should be d ** n, " \
            "where d is subsystem Hilbert space dimension and n is the number of subsystems. " \
            "Amplitude length: {}, expected subsystems: {}, num keys: {}".format(
                len(amplitudes), num_subsystems, len(keys)
            )

        self.state = array(amplitudes, dtype=complex)
        self.keys = keys


class DensityState(State):
    """Class to represent an individual quantum state as a density matrix.

    Attributes:
        state (np.array): density matrix values. NxN matrix with N = d ** len(keys), where d is dimension of elementary
            Hilbert space. Default is d = 2 for qubits.
        keys (List[int]): list of keys (subsystems) associated with this state.
        truncation (int): maximally allowed number of excited states for elementary subsystems.
            Default is 1 for qubit. dim = truncation + 1
    """

    def __init__(self, state: List[List[complex]], keys: List[int], truncation: int = 1):
        """Constructor for density state class.

        Args:
            state (List[List[complex]]): density matrix elements given as a list.
                If the list is one-dimensional, will be converted to matrix with outer product operation.
            keys (List[int]): list of keys to this state in quantum manager.
            truncation (int): maximally allowed number of excited states for elementary subsystems.
                Default is 1 for qubit. dim = truncation + 1
        """

        super().__init__()
        self.truncation = truncation
        dim = self.truncation + 1  # dimension of element Hilbert space

        # print(state)
        state = array(state, dtype=complex)
        if state.ndim == 1:
            state = outer(state, state.conj())

        # check formatting
        assert abs(trace(array(state)) - 1) < 0.01, "density matrix trace must be 1"
        for row in state:
            assert len(state) == len(row), "density matrix must be square"

        num_subsystems = log(len(state)) / log(dim)
        assert dim ** int(round(num_subsystems)) == len(state), \
            "Length of amplitudes should be d ** n, " \
            "where d is subsystem Hilbert space dimension and n is the number of subsystems. " \
            "Actual amplitude length: {}, dim: {}, num subsystems: {}".format(
                len(state), dim, num_subsystems
            )
        num_subsystems = int(round(num_subsystems))
        assert num_subsystems == len(keys), \
            "Length of amplitudes should be d ** n, " \
            "where d is subsystem Hilbert space dimension and n is the number of subsystems. " \
            "Amplitude length: {}, expected subsystems: {}, num keys: {}".format(
                len(state), num_subsystems, len(keys)
            )

        self.state = state
        self.keys = keys


# Not using this.
class DensityState_sparse(State):
    """Class to represent an individual quantum state as a density matrix.

    Attributes:
        state (np.array): density matrix values. NxN matrix with N = d ** len(keys), where d is dimension of elementary
            Hilbert space. Default is d = 2 for qubits.
        keys (List[int]): list of keys (subsystems) associated with this state.
        truncation (int): maximally allowed number of excited states for elementary subsystems.
            Default is 1 for qubit. dim = truncation + 1
    """

    def __init__(self, state: List[List[complex]], keys: List[int], truncation: int = 1):
        """Constructor for density state class.

        Args:
            state (List[List[complex]]): density matrix elements given as a list.
                If the list is one-dimensional, will be converted to matrix with outer product operation.
            keys (List[int]): list of keys to this state in quantum manager.
            truncation (int): maximally allowed number of excited states for elementary subsystems.
                Default is 1 for qubit. dim = truncation + 1
        """

        super().__init__()
        self.truncation = truncation
        dim = self.truncation + 1  # dimension of element Hilbert space


        state = sp.csr_array(state, dtype=complex)
        # print(state.toarray())
        if state.shape[0] == 1:
            # state = outer(state, state.conj())
            state = state.T.conj() * state

        # check formatting
        assert abs(state.trace() - 1) < 0.01, "density matrix trace must be 1"
        for row in state:
            # print("row shape:", row.shape[1])
            assert state.shape[0] == row.shape[1], "density matrix must be square"

        num_subsystems = log(state.shape[0]) / log(dim)
        assert dim ** int(round(num_subsystems)) == state.shape[0], \
            "Length of amplitudes should be d ** n, " \
            "where d is subsystem Hilbert space dimension and n is the number of subsystems. " \
            "Actual amplitude length: {}, dim: {}, num subsystems: {}".format(
                state.shape[0], dim, num_subsystems
            )
        num_subsystems = int(round(num_subsystems))
        assert num_subsystems == len(keys), \
            "Length of amplitudes should be d ** n, " \
            "where d is subsystem Hilbert space dimension and n is the number of subsystems. " \
            "Amplitude length: {}, expected subsystems: {}, num keys: {}".format(
                state.shape[0], num_subsystems, len(keys)
            )

        self.state = state
        self.keys = keys


class FreeQuantumState(State):
    """Class used by photons to track internal quantum states.

    This is an alternative to tracking states in a dedicated quantum manager, which adds simulation overhead.
    It defines several operations, including entanglement and measurement.
    For memories with an internal quantum state and certain photons, such as those stored in a memory or in parallel
    simulation, this class should not be used.
    Quantum states stored in a quantum manager class should be used instead.
    This module uses the ket vector formalism for storing and manipulating states.

    Attributes:
        state (Tuple[complex]): list of complex coefficients in Z-basis.
        entangled_states (List[QuantumState]): list of entangled states (including self).
    """

    def __init__(self):
        super().__init__()
        self.state = (complex(1), complex(0))
        self.entangled_states = [self]
        self.density_matrix = False

    def combine_state(self, another_state: "FreeQuantumState"):
        """Method to tensor multiply two quantum states.

        Arguments:
            another_state (QuantumState): state to entangle current state with.

        Side Effects:
            Modifies the `entangled_states` field for current state and `another_state`.
            Modifies the `state` field for current state and `another_state`.
        """

        entangled_states = self.entangled_states + another_state.entangled_states
        new_state = kron(self.state, another_state.state)
        new_state = tuple(new_state)

        for quantum_state in entangled_states:
            quantum_state.entangled_states = entangled_states
            quantum_state.state = new_state

    # def split_states(self):
    #     input_state = self.state
    #     num_systems = int(log2(len(input_state)))
    #     all_indices = list(range(num_systems)) 
    #     tuple_state = tuple(map(tuple, outer(input_state, input_state.conj())))

    #     for i in range(num_systems): # you have log2(len(state)) qubits in the quantum state. You will need logd(state) for qudits.
    #         self.entangled_states[i].state = (density_partial_trace(tuple_state, tuple(all_indices[:i]+all_indices[i+1:]), num_systems))
    #         self.entangled_states[i].entangled_states -= self.entangled_states

    # def split_states(self):
    #     input_state = self.state
    #     num_systems = int(log2(len(input_state)))
    #     all_indices = list(range(num_systems)) 
    #     tuple_state = tuple(map(tuple, outer(input_state, input_state.conj())))

    #     index = self.entangled_states.index(self)

    #     for i in all_indices[:index]+all_indices[index+1:]:
    #         self.entangled_states[i].state = density_partial_trace(tuple_state, (index,), num_systems)    

    #     self.entangled_states.remove(self)
    #     self.entangled_states = [self]
    #     self.state = (density_partial_trace(tuple_state, tuple(all_indices[:index]+all_indices[index+1:]), num_systems))

    def split_states(self):
        input_state = np.array(self.state)
        num_systems = int(log2(len(input_state))) # We use log base 2 here, for qubits. That may not be the case in the case of general photons. 
        all_indices = list(range(num_systems)) 
        if not self.density_matrix:
            tuple_state = tuple(map(tuple, outer(input_state, input_state.conj())))
            self.density_matrix = True
        else:
            tuple_state = tuple(map(tuple, input_state))

        index = self.entangled_states.index(self)

        new_state = tuple(map(tuple, density_partial_trace(tuple_state, (index,), num_systems)))

        for i in all_indices[:index]+all_indices[index+1:]:
            self.entangled_states[i].state = new_state
            self.entangled_states[i].density_matrix = True

        self.entangled_states.remove(self)
        self.entangled_states = [self]
        self.state = tuple(map(tuple, density_partial_trace(tuple_state, tuple(all_indices[:index]+all_indices[index+1:]), num_systems)))
        # self.density_matrix = True


    def build_kraus_ops(self, dim):
        num_qubits = np.log2(dim)
        kraus_ops = [None]*4
        kraus_ops[0] = np.sqrt(1-3*self.polarization_fidelity/4) * np.array([[1,0], [0,1]])
        kraus_ops[1] = np.sqrt(self.polarization_fidelity/4) * np.array([[0,1], [1,0]])
        kraus_ops[2] = np.sqrt(self.polarization_fidelity/4) * np.array([[0,-1j], [1j,0]])
        kraus_ops[3] = np.sqrt(self.polarization_fidelity/4) * np.array([[1,0], [0,-1]])
        return kraus_ops


    def random_noise(self, rng: Generator):
        """Method to add random noise to a single state.

        Chooses a random angle to set the quantum state to (with no phase difference).

        Side Effects:
            Modifies the `state` field.
        """

        # TODO: rewrite for entangled states
        if self.density_matrix:
            dim = len(self.state)
            kraus_ops = self.build_kraus_ops(dim)
            final_state = 0
            for kraus_op1 in kraus_ops:
                for kraus_op2 in kraus_ops:
                    kraus_op = np.kron(kraus_op1, kraus_op2)
                    final_state += kraus_op @ self.state @ kraus_op.conj()
            self.state = final_state
        else:
            angle = rng.random() * 2 * pi
            self.state = (complex(cos(angle)), complex(sin(angle)))

    # only for use with entangled state
    def set_state(self, state: Tuple[complex], density_matrix = False):
        """Method to change entangled state of multiple quantum states.

        Args:
            state (Tuple[complex]): new coefficients for state.
                Should be 2^n in length, where n is the length of `entangled_states`.

        Side Effects:
            Modifies the `state` field for current and entangled states.
        """

        # check formatting of state
        if density_matrix:
            assert all([abs(state[i,i]) <= 1.01 for i in range(len(state))]), "Illegal value with abs > 1 in quantum state"
        else:
            assert all([abs(a) <= 1.01 for a in state]), "Illegal value with abs > 1 in quantum state"
        # assert abs(sum([abs(a) ** 2 for a in state]) - 1) < 1e-5, "Squared amplitudes do not sum to 1"

        num_qubits = log2(len(state))
        assert 2 ** int(round(num_qubits)) == len(state),"Length of amplitudes should be 2 ** n, where n is the number of qubits. Actual amplitude length: {}, num qubits: {}".format(len(state), num_qubits)
        
        num_qubits = int(round(num_qubits))
        assert num_qubits == len(self.entangled_states), "Length of amplitudes should be 2 ** n, where n is the number of qubits. Num qubits in state: {}, num qubits in object: {}".format(num_qubits, len(self.entangled_states))

        for qs in self.entangled_states:
            qs.state = state
            qs.density_matrix = density_matrix 
        self.density_matrix = density_matrix 
    # for use with single, unentangled state
    def set_state_single(self, state: Tuple[complex]):
        """Method to unentangle and set the state of a single quantum state object.

        Args:
            state (Tuple[complex]): 2-element list of new complex coefficients.

        Side Effects:
            Will remove current state from any entangled states (if present).
            Modifies the `state` field of current state.
        """

        for qs in self.entangled_states:
            if qs is not None and qs != self:
                index = qs.entangled_states.index(self)
                qs.entangled_states[index] = None
        self.entangled_states = [self]
        self.state = state

    def measure(self, basis: Tuple[Tuple[complex]], rng: Generator, return_prob = False) -> int:
        """Method to measure a single quantum state.

        Args:
            basis (Tuple[Tuple[complex]]): measurement basis, given as list of states
                (that are themselves lists of complex coefficients).
            rng (Generator): random number generator for measurement

        Returns:
            int: 0/1 measurement result, corresponding to one basis vector.

        Side Effects:
            Modifies the `state` field for current and any entangled states.
        """

        # handle entangled case
        if self.density_matrix:
            self.state = tuple(map(tuple, self.state))
        if len(self.entangled_states) > 1:
            num_states = len(self.entangled_states)
            state_index = self.entangled_states.index(self)
            # state0, state1, prob = measure_entangled_state_with_cache(self.state, basis, state_index, num_states)
            states, probs = measure_multiple_with_cache_density(self.state, 1, 1)
            if return_prob:
                return states, probs
            # print("prob:", prob)
            if rng.random() < probs[0]:
                new_state = states[0]
                result = 0
            else:
                new_state = states[1]
                result = 1
            new_state = tuple(new_state)

        # handle unentangled case
        else:
            prob = measure_state_with_cache(self.state, basis)
            # print("prob is:", np.round(prob, 4))
            # print("prob:", prob)

            if return_prob:
                return prob

            if rng.random() < prob:
                new_state = basis[0]
                result = 0
            else:
                new_state = basis[1]
                result = 1

        # set new state
        # new_state = tuple(new_state)
        for s in self.entangled_states:
            if s is not None:
                s.state = new_state

        return result

    @staticmethod
    def measure_multiple(basis, states, rng: Generator, return_states = False):
        """Method to measure multiple qubits in a more complex basis.

        May be used for bell state measurement.

        Args:
            basis (List[List[complex]]): list of basis vectors.
            states (List[QuantumState]): list of quantum state objects to measure.
            rng (Generator): random number generator for measurement

        Returns:
            int: measurement result in given basis.

        Side Effects:
            Will modify the `state` field of all entangled states.
        """

        # ensure states are entangled
        # (must be entangled prior to calling measure_multiple)
        entangled_list = states[0].entangled_states
        # print("all entangled states:", type(states[0].entangled_states[0]))
        for state in states[1:]:
            # print("sent states:", type(state))
            assert state in entangled_list
        # ensure basis and vectors in basis are the right size
        basis_dimension = 2 ** len(states)
        assert len(basis) == basis_dimension
        for vector in basis:
            assert len(vector) == len(basis)

        state = states[0].state

        #################### Omiting this for now since the experiment only entangles 2 qubits at max at any time. #######
        # # move states to beginning of entangled list and quantum state
        # pos_state_0 = entangled_list.index(states[0])
        # pos_state_1 = entangled_list.index(states[1])
        # entangled_list[0], entangled_list[pos_state_0] = entangled_list[pos_state_0], entangled_list[0]
        # entangled_list[1], entangled_list[pos_state_1] = entangled_list[pos_state_1], entangled_list[1]
        # switched_state = [complex(0)] * len(state)
        # for i, coefficient in enumerate(state):
        #     switched_i = swap_bits(i, pos_state_0, pos_state_1)
        #     switched_state[switched_i] = coefficient
        # state = tuple(map(tuple, switched_state))

        state = tuple(map(tuple, state))

        # math for probability calculations
        length_diff = len(entangled_list) - len(states)

        # new_states, probabilities = measure_multiple_with_cache(state, basis, length_diff)
        new_states, probabilities = measure_multiple_with_cache_density(state, len(states), length_diff)
        if return_states:
            return new_states, probabilities

        possible_results = arange(0, basis_dimension, 1)
        # result gives index of the basis vector that will be projected to
        res = rng.choice(possible_results, p=probabilities)
        # project to new state, then reassign quantum state and entangled photons
        new_state = new_states[res]
        for state in entangled_list:
            state.quantum_state = new_state
            state.entangled_photons = entangled_list

        return res
