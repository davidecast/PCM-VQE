# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:24:04 2021

@author: castd
"""

import pennylane as pl
import pennylane.numpy as np
import numpy as np_original
from openfermion import FermionOperator
from pennylane_qchem.qchem.structure import convert_observable
from pennylane_qchem.qchem.obs import one_particle
from openfermion.ops.operators import QubitOperator
import pennylane_qchem as plq
from openfermion.transforms import jordan_wigner_one_body
from openfermion.transforms import jordan_wigner
from functools import reduce
from operator import mul
from itertools import product

def set_1_rdm(num_so):
    
   # rdm1 = np.zeros((num_so, num_so))
    rdm1 = list() 
    for i in range(num_so):
        for j in range(num_so):
            rdm1.append(FermionOperator(str(i)+ '^ ' + str(j), 1.0))
   # rdm1 = reshape(rdm1, [num_so, num_so])
   # print(np.shape(rdm1))
    return rdm1

def set_1_rdm_via_one_particle(num_so, core = None, active = None):
    rdm_matrix = np.ones((num_so // 2, num_so // 2))
#    fermionic = one_particle(np.ones((num_so // 2, num_so // 2)), core = core, active = active)
#    rdm1 = list()
#    for element in fermionic:
        
        

def set_1_rdm_via_qubit_operator(num_so):
    rdm1 = list()
    for i in range(num_so // 2):
        for j in range(num_so // 2):
            if i >= j:
                rdm1.append(jordan_wigner_one_body(2 * i, 2 * j) + jordan_wigner_one_body(2 * i + 1, 2 * j + 1))
    return rdm1


def density_matrix_to_array(list_to_transform):
    array_shape = np.shape(list_to_transform)
    transformed_array = np.zeros(tuple(array_shape))
    for element, _ in np.ndenumerate(transformed_array):
        transformed_array[element] = list_to_transform[element[0]][element[1]]
    return transformed_array


def set_2_rdm(num_so):
    
   # rdm2 = np.zeros((num_so, num_so, num_so, num_so))
    rdm2 = list()
    for i in range(num_so):
        for j in range(num_so):
            for k in range(num_so):
                for l in range(num_so):
                    rdm2.append(FermionOperator(str(i)+ '^ ' + str(j) + '^ ' +str(k)+ ' ' + str(l), 1.0))
 #   rdm2 = reshape(rdm2, [num_so, num_so, num_so, num_so])                
    return rdm2

def from_list_to_array_rdm_mo(num_so, list_to_transform):
    rdm_array = reshape([0] * (num_so // 2)**2, [num_so // 2, num_so // 2])
    counter_to_pick_element = 0
    for i in range(num_so // 2):
        for j in range(num_so // 2):
            if i > j:
                rdm_array[i][j] = 0.5 * list_to_transform[counter_to_pick_element]
                rdm_array[j][i] = rdm_array[i][j]
                counter_to_pick_element += 1
            elif i == j:
                rdm_array[i][i] = list_to_transform[counter_to_pick_element]
                counter_to_pick_element += 1
    return rdm_array

def trace_over_spin(num_orbitals, one_srdm):  #### attention this should follow qiskit convention (adadpted from below) 
    n_orbital = num_orbitals
    one_rdm = np.zeros(tuple([n_orbital]*2))
    for mol_coords, _ in np.ndenumerate(one_rdm):

        rdm_contribution = 0.
        one_rdm[mol_coords] = 0.0
        for spin_coords in product([mol_coords[0], mol_coords[0] + 2 * num_orbitals // 2], [mol_coords[1], mol_coords[1] + 2 * num_orbitals // 2]):

           # coeff = -1. if (spin_coords[0] // n_orbital != spin_coords[1] // n_orbital) else 1.  #### questo è per orbitali complessi (?)
            coeff = 1. if (spin_coords[0] // n_orbital != spin_coords[1] // n_orbital) else 1.
            i,j = spin_coords
            if isinstance(one_srdm[i][j], float):
                rdm_contribution += coeff * one_srdm[i][j]
            else:
                rdm_contribution += coeff * one_srdm[i][j]._value
       # print(rdm_contribution)
        one_rdm[mol_coords] = rdm_contribution
    return one_rdm

def spatial_from_spinorb(num_orbitals, one_srdm):
    n_qubits = 2 * num_orbitals
    one_rdm = np.zeros(tuple([num_orbitals]*2))
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):
            one_rdm[p, q] = one_srdm[2 * p][ 2 * q]
            one_rdm[p, q] += one_srdm[2 * p + 1][ 2 * q + 1]
    return one_rdm

#### ---- Possible option to isolate differente contributions in the VQE procedure ---- #### (From OpenQemist implementation)

def get_rdm_masks(num_orbitals): ### for spinless rdms
    """
        Obtain the 1- and 2-RDM matrices for given variational parameters.
        This makes sense for problem decomposition methods if these amplitudes are the ones
        that minimizes energy.

    Returns:
        (numpy.array, numpy.array): One & two-particle RDMs (rdm1_np & rdm2_np, float64).
    Raises:
        RuntimeError: If no simulation has been run before calling this method.
    """

    # Initialize RDM matrices and other work arrays
    n_orbital = num_orbitals
    one_rdm = np.zeros(tuple([n_orbital]*2))
    two_rdm = np.zeros(tuple([n_orbital]*4))

#    tmp_h1 = np.zeros(self.one_body_integrals.shape)
#    tmp_h2 = np.zeros(self.two_body_integrals.shape)

    # h1 and h2 are the one- and two-body integrals for the whole system
    # They are in spin-orbital basis, seemingly with all alpha orbitals first and then all beta second
    # eg lines and columns refer to 1a, 2a, 3a ... , then 1b, 2b, 3b ....

    # Compute values for 1-RDM matrix
    # -------------------------------
    for mol_coords, _ in np.ndenumerate(one_rdm):

        rdm_contribution = 0.
        one_rdm[mol_coords] = 0.0

        # Find all entries of one-RDM that contributes to the computation of that entry of one-rdm
        for spin_coords in product([mol_coords[0], mol_coords[0] + 2 * num_orbitals // 2],
                                   [mol_coords[1], mol_coords[1] + 2 * num_orbitals // 2]):

            # Skip values too close to zero
#            if abs(one_body_integrals[spin_coords]) < 1e-10:
#                continue

            # Ignore all Fermionic Hamiltonian term except one
#            tmp_h1[spin_coords] = 1.
            coeff = -1. if (spin_coords[0] // n_orbital != spin_coords[1] // n_orbital) else 1. ### we consider wavefunctions that can be complexes

        # Write the value to the 1-RDM matrix
        one_rdm[mol_coords] = coeff

    # Compute values for 2-RDM matrix
    # -------------------------------
    for mol_coords, _ in np.ndenumerate(two_rdm):

        rdm_contribution = 0.
        two_rdm[mol_coords] = 0.0

        # Find all entries of h1 that contributes to the computation of that entry of one-rdm
        for spin_coords in product([mol_coords[0], mol_coords[0] + 2 * num_orbitals // 2],
                                   [mol_coords[1], mol_coords[1] + 2 * num_orbitals // 2],
                                   [mol_coords[2], mol_coords[2] + 2 * num_orbitals // 2],
                                   [mol_coords[3], mol_coords[3] + 2 * num_orbitals // 2]):

            # Skip values too close to zero
#            if abs(two_body_integrals[spin_coords]) < 1e-10:
#                continue

            # Set entries to the right coefficient for tmp_h1
#            tmp_h2[spin_coords] = 1.

            # Count alphas and betas. If odd, coefficient is -1, else its 1.

            n_betas_total = sum([spin_orb // n_orbital for spin_orb in spin_coords])
            if (n_betas_total == 0) or (n_betas_total == 4):
                coeff = 2.0
            elif n_betas_total == 2:
                coeff = -1.0 if (spin_coords[0] // n_orbital != spin_coords[1] // n_orbital) else 1.0

            rdm_contribution += coeff 

            # Reset entries of tmp_h2
#            tmp_h2[spin_coords] = 0.

        # Write the value to the 1-RDM matrix
        two_rdm[mol_coords] = rdm_contribution

    return one_rdm, two_rdm


def set_1_rdm_coefficients_corrected(num_so, mask):  ### this works in the MO basis 

    rdm1 = np.zeros((num_so, num_so))
    
    for i in range(num_so):
        for j in range(num_so):
            rdm1[i][j] = FermionOperator('i^ j', mask[i][j])
            
    return rdm1


def set_2_rdm_coefficients_corrected(num_so, mask): ### this works in the MO basis
    
    rdm2 = np.zeros((num_so, num_so, num_so, num_so))
    
    for i in range(num_so):
        for j in range(num_so):
            for k in range(num_so):
                for l in range(num_so):
                    rdm2[i][j][k][l] = FermionOperator('i^ j^ k l', mask[i][j][k][l])
                    
    return rdm2    

def penny_lane_version_one_particle_operator(mask): ### also this should work in the MO basis
    return plq.qchem.one_particle(mask)
    

def build_observable_for_rdm(fermionic_operator_list, mapping):
#    if mapping == 'jordan_wigner':
#        qubit_operator = jordan_wigner(fermionic_operator_list)
    observable_elements_rdm = list(map(convert_observable, fermionic_operator_list))
   # observable_elements_rdm = [list(map(plq.qchem.obs.observable, sublist[:])) for sublist in fermionic_operator_list]
    return observable_elements_rdm

def reshape(lst, shape):
    if len(shape) == 1:
        return lst
    n = reduce(mul, shape[1:])
    return [reshape(lst[i*n:(i+1)*n], shape[1:]) for i in range(len(lst)//n)]
    
 ########## ------------------------ ###########   

            
    
