# -*- coding: utf-8 -*-

import pennylane as pl
import pennylane.numpy as np
import numpy as np_original
import density_matrix_helper
import psi4
from qiskit import IBMQ
from qiskit.test.mock import FakeMumbai
from qiskit.providers.aer.noise import NoiseModel
from openfermion.chem import molecular_data
from pennylane import qchem
from functools import partial
from pennylane.templates import UCCSD
from pennylane_qchem.qchem.structure import convert_observable
import openfermion.ops.representations as reps
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.ops import general_basis_change
##### la classe seguente sicuramente deve avere un attributo psi4 di tipo wfn perchè dobbiamo poter accedere 
##### alla funzione di update dell'energia di polarizzazione di PCMSolver

##### Deve potersi creare il molecular_hamiltonian a partire dagli integrali!! (sempre accessibili tramite wfn)

##### in realtà non per forza possiamo anche costruire le rdms e basta

##### Al momento facciamo tutto con le matrici densita 1,2 rdm però questo non dovrebbe essere la scelta migliore
##### dal punto di vista computazionale (perchè non parallelizziamo niente) ma abbiamo più controllo su ipotetici 
##### metodi di error mitigation

def noise_model():
    device_backend = FakeMumbai()
    noise_model = NoiseModel.from_backend(device_backend)
    return noise_model
    

class PCM_VQE():
    
    def __init__(self, psi4_wavefunction, vqe_options, molecule_options = None):
        self.wfn = psi4_wavefunction ### psi4_wfn object to update the matrices
        self.options = vqe_options  ### dictionary containing all the possible informations about the calculation
        self.molecule_options = molecule_options    ### when a geometry is given the standard PL workflow is executed (is the real "/path/to/file.xyz")
        self.rdm1_obs = None
        self.rdm2_obs = None
        self.save_rdm = False
        self.transformed_molecular_integrals_one_body = None
        self.transformed_molecular_integrals_two_body = None
        self.circuit = None
        self.dev = None
        self.molecular_hamiltonian = None
        self.polarization_energy = list()
        self.num_estimated_electrons = list()
        self.density_matrices = list()
        self.opt = None
        self.qubits = None

    
    def transform_integrals(self): #### OpenFermion convention is to map orbital 0 with spin up to qubit 0
                                   #### ------------------------------- orbital 0 with spin down to qubit 1
        if self.molecule_options != None:
            symbols, coordinates = qchem.read_structure(self.molecule_options['structure'])
            H, qubits = qchem.molecular_hamiltonian(symbols, coordinates, charge = self.molecule_options['charge'],
                                                    package = 'psi4', mapping= self.options['mapping'], active_electrons = self.molecule_options['active_electrons'])
            #H = pl.utils.sparse_hamiltonian(H)
            self.molecular_hamiltonian , self.qubits = H, qubits
        else:
            C = np.asarray(self.wfn.Ca())
            mints = psi4.core.MintsHelper(self.wfn.basisset())
            H = general_basis_change(np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential()), C, (1, 0))
            mo_eri = np.asarray(mints.ao_eri())
            mo_eri.reshape((self.wfn.nmo(), self.wfn.nmo(), self.wfn.nmo(), self.wfn.nmo()))
            mo_eri = np.einsum('psqr', mo_eri)
            mo_eri = general_basis_change(mo_eri, C, (1, 1, 0, 0))
            self.transformed_molecular_integrals_one_body, self.transformed_molecular_integrals_two_body = molecular_data.spinorb_from_spatial(H ,mo_eri)
            if self.save_rdm:
                self.dipole_ints = []
                dipole_integrals = np.asarray(mints.ao_dipole())
                for k in range(len(dipole_integrals)):
                    dipole_integrals[k] = general_basis_change(dipole_integrals[k], C, (1, 0))
                    self.dipole_ints.append(molecular_data.spinorb_from_spatial(dipole_integrals[k]))
        
    
    def build_rdms_observable(self):
        if self.qubits != None:
            rdm1_qubit_operators = density_matrix_helper.set_1_rdm_via_qubit_operator(self.qubits)
        else:
            rdm1_qubit_operators = density_matrix_helper.set_1_rdm_via_qubit_operator(2 * self.wfn.nmo())
        rdm1_observable = density_matrix_helper.build_observable_for_rdm(rdm1_qubit_operators, self.options['mapping'])
        if self.qubits != None:
            rdm1_observable = density_matrix_helper.from_list_to_array_rdm_mo(self.qubits, rdm1_observable)
        else:
            rdm1_observable = density_matrix_helper.from_list_to_array_rdm_mo(2 * self.wfn.nmo(), rdm1_observable)
        self.rdm1_obs = rdm1_observable

    
    def compute_polarization_energy(self, spinless_one_electron):
        if self.options['simulation'] != 'statevector':
            estimated_electrons = np.trace(spinless_one_electron)
            self.num_estimated_electrons.append(estimated_electrons)
            self.density_matrices.append(spinless_one_electron)
         #   spinless_one_electron *= self.molecule_options['active_electrons']/estimated_electrons
         #   self.density_matrices.append(spinless_one_electron)
        else:
            self.density_matrices.append(spinless_one_electron)
     #   if self.molecule_options['core_orbitals'] != 0:
        if self.molecule_options != None:
            density_matrix = density_matrix_helper.density_matrix_to_array(spinless_one_electron)
            if self.options['simulation'] == 'simulated_quantum_noise':
                density_matrix *= self.molecule_options['active_electrons']/estimated_electrons
            self.density_matrices.append(density_matrix)
            orlo =  np_original.eye(self.molecule_options['core_orbitals']) * 2
            orlated_rdm_mo = np_original.block([[orlo, np_original.zeros((self.molecule_options['core_orbitals'], self.qubits // 2))],
                                        [np_original.zeros((self.qubits // 2, self.molecule_options['core_orbitals'])), density_matrix]])
            rdm1_ao_basis = general_basis_change(orlated_rdm_mo, np.transpose(np.asarray(self.wfn.Ca())), (1, 0))
        else:
            density_matrix = density_matrix_helper.density_matrix_to_array(spinless_one_electron)
            if self.options['simulation'] == 'simulated_quantum_noise':
                density_matrix *= self.molecule_options['active_electrons']/estimated_electrons
            self.density_matrices.append(density_matrix)
            rdm1_ao_basis = general_basis_change(density_matrix, np.transpose(np.asarray(self.wfn.Ca())), (1, 0))
        polarization_energy , _ = self.wfn.get_PCM().compute_PCM_terms(psi4.core.Matrix(self.wfn.nmo(), self.wfn.nmo()).from_array(rdm1_ao_basis), psi4.core.PCM.CalcType.Total)
        self.polarization_energy.append(polarization_energy)
        #print(polarization_energy)
        return polarization_energy
    
    
    def cost_function(self, params):
        one_electron_rdm = self.compute_1rdm(params)
        if self.save_rdm:  ##### correct usage only for dipole moment calculation (see run file example)
            self.density_matrices.append(one_electron_rdm)
        cost_function = self.compute_electronic_contribution(params) #vacuum
        #cost_function = self.compute_electronic_contribution_sparse(params) #vacuum
        #print(cost_function)
        if self.options['PCM'] == True:
            cost_function += self.compute_polarization_energy(one_electron_rdm)
      #  cost_function += self.wfn.variable("Nuclear Repulsion Energy")
        #print(cost_function)
        return cost_function

    def compute_dipole_moment(self, params):
        return [pl.vqe.ExpvalCost(self.circuit, self.dipole_operator[0], self.dev, optimize = True)(params),
                pl.vqe.ExpvalCost(self.circuit, self.dipole_operator[1], self.dev, optimize = True)(params), 
                pl.vqe.ExpvalCost(self.circuit, self.dipole_operator[2], self.dev, optimize = True)(params)]
    
    def compute_electronic_contribution(self, params):
        #pl.vqe.ExpvalCost(pl.SparseHamiltonian(self.molecular_hamiltonian, wires=range(self.qubits)), optimize = True)(params)
        return pl.vqe.ExpvalCost(self.circuit, self.molecular_hamiltonian, self.dev, optimize = True)(params)

    def compute_electronic_contribution_sparse(self, params):
        return circuit_sparse_evaluation(params, self.molecular_hamiltonian, self.qubits)
    
    def compute_one_electron_contribution(self, one_electron_rdm):
        return np.sum(np.multiply(self.transformed_molecular_integrals_one_body, one_electron_rdm))
    
    def compute_two_electron_contribution(self, two_electron_rdm):
        return np.sum(np.multiply(self.transformed_molecular_integrals_two_body, two_electron_rdm))
    
    def compute_1rdm(self, params):
        one_electron_rdm = list()
        if self.qubits != None:
            for i in range(self.qubits // 2):
                for j in range(self.qubits // 2):
                    one_electron_rdm.append(pl.vqe.ExpvalCost(self.circuit, self.rdm1_obs[i][j], self.dev)(params))
            one_electron_rdm = density_matrix_helper.reshape(one_electron_rdm, [self.qubits // 2, self.qubits // 2])
        else:
            for i in range(self.wfn.nmo()):
                for j in range(self.wfn.nmo()):
                    one_electron_rdm.append(pl.vqe.ExpvalCost(self.circuit, self.rdm1_obs[i][j], self.dev)(params))
            one_electron_rdm = density_matrix_helper.reshape(one_electron_rdm, [self.wfn.nmo(), self.wfn.nmo()])
        return one_electron_rdm

    def compute_1rdm_mo(self, params):
        one_electron_rdm = [pl.vqe.ExpvalCost(self.circuit, element, self.dev)(params) for element in self.rdm1_obs]
        one_electron_rdm = density_matrix_helper.reshape(one_electron_rdm, [self.wfn.nmo(), self.wfn.nmo()])
        return one_electron_rdm

    
    def set_openfermion_molecular_hamiltonian_jordan_wigner(self):
        if self.qubits == None:
            molecular_hamiltonian = reps.InteractionOperator(self.wfn.variable("Nuclear Repulsion Energy"), self.transformed_molecular_integrals_one_body, 1 / 2 * self.transformed_molecular_integrals_two_body)
            fermionic_hamiltonian = get_fermion_operator(molecular_hamiltonian)
            transformed_to_qubit = jordan_wigner(fermionic_hamiltonian)
            self.molecular_hamiltonian = convert_observable(transformed_to_qubit)
            if self.save_rdm:
                self.dipole_operator = []
                for k in range(len(self.dipole_ints)):
                    dipole_operator = reps.InteractionOperator(self.dipole_ints[k])
                    fermionic_dipole = get_fermion_operator(dipole_operator)
                    dipole_to_qubit = jordan_wigner(fermionic_dipole)
                    self.dipole_operator.append(convert_observable(dipole_to_qubit))
        else:
            pass


    def compute_2rdm(self, params):
        two_electron_rdm = list()
        for i in range(2 * self.wfn.nmo()):
            for j in range(2 * self.wfn.nmo()):
                for k in range(2 * self.wfn.nmo()):
                    for l in range(2 * self.wfn.nmo()):
                        two_electron_rdm.append(pl.vqe.ExpvalCost(self.circuit, self.rdm2_obs[i][j][k][l], self.dev)(params))
        two_electron_rdm = density_matrix_helper.reshape(two_electron_rdm, [2 * self.wfn.nmo(), 2 * self.wfn.nmo(), 2 * self.wfn.nmo(), 2 * self.wfn.nmo()])
        return two_electron_rdm
    
    def set_up_ansatz_and_guess(self):
        if self.qubits != None:
            electrons = self.molecule_options['active_electrons'] 
            qubits = self.qubits 
        else:
            electrons = 2 * self.wfn.nalpha()  ### stiamo considerando HF states di singoletto
            qubits = 2 * self.wfn.nmo() ### we are considering only jw mapping for simplicity
        if self.options['simulation'] == "statevector":
            self.dev = pl.device('default.qubit', wires=qubits)
        elif self.options['simulation'] == "sampling_noise":
            pass
        elif self.options['simulation'] == "simulated_quantum_noise":
            self.dev = pl.device('qiskit.aer', wires=qubits, noise_model=noise_model(), shots = 8192)
        elif self.options['simulation'] == "qpu":
            pass
        if self.options['ansatz'] == 'UCCSD':
            ref_state = qchem.hf_state(electrons, qubits)
            singles, doubles = qchem.excitations(electrons, qubits)
            s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)  ### try to modify with single and double excitations templates
            self.circuit = partial(UCCSD, init_state=ref_state, s_wires=s_wires, d_wires=d_wires)
            self.initial_guess = np.random.normal(0, np.pi, len(singles) + len(doubles))
        else:
            self.circuit = partial(self.custom_circuit, wires = qubits)
            #self.initial_guess = np.random.normal(0, np.pi, 2)
            self.initial_guess = np.array([0,0])
            
    def custom_circuit(self, params, wires):
        hf = np.array([1, 1, 0, 0, 0, 0])
        pl.BasisState(hf, wires=wires)
        pl.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
        pl.DoubleExcitation(params[1], wires=[0, 1, 4, 5])
    

    def grad_cost_fn(self, params):
        grad_h = pl.finite_diff(self.cost_function)(params)
        return grad_h
        
    def set_optimizer(self):
        #self.opt = pl.RotosolveOptimizer()
        self.opt = pl.AdagradOptimizer(stepsize = 0.1)
        
    def run(self):
        params = self.initial_guess
        result = {'energy': [], 'convergence': [], 'params': [], 'polarization_energy': self.polarization_energy, 'estimated_electrons': self.num_estimated_electrons, 'density_matrix': self.density_matrices}
        for n in range(self.options['max_iterations']):
            params, prev_energy = self.opt.step_and_cost(self.cost_function, params, grad_fn = self.grad_cost_fn)
           # params, prev_energy = self.opt.step_and_cost(self.cost_function, params)
            energy = self.cost_function(params)
            conv = np.abs(energy - prev_energy)
            result['energy'].append(energy)
            result['convergence'].append(conv)
            if n % 1 == 0:
                print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, energy))
            if n % 20 == 0:
                result['params'].append(params)
            #if conv <= self.options['conv_tol']:
            #    break
        print('Final value of the ground-state energy = {:.8f} Ha'.format(energy))
        if self.save_rdm:
            print('Computing dipole moment ...')
            dipole_moment = self.compute_dipole_moment(params)
            print(dipole_moment)
        #print('Accuracy with respect to the CCSD-PCM energy: {:.8f} Ha ({:.8f} kcal/mol)'.format(np.abs(energy - (-1.137391597913367)), np.abs(energy - (-1.137391597913367))*627.503))
        return result
    
#@pl.qnode(pl.device('qiskit.aer', wires=6, noise_model=noise_model()), diff_method = "parameter-shift")
@pl.qnode(pl.device('default.qubit', wires = 6), diff_method = "parameter-shift")
def circuit_sparse_evaluation(params, hamiltonian, qubits):
    hf = np.array([1, 1, 0, 0, 0, 0])
    pl.BasisState(hf, wires=range(qubits))
    pl.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
    pl.DoubleExcitation(params[1], wires=[0, 1, 4, 5])
    return pl.expval(pl.SparseHamiltonian(hamiltonian, wires=range(qubits)))

    

                        
    
