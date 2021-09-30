# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 16:12:50 2021

@author: castd
"""
import time
from vqe_pcm_pennylane import PCM_VQE
import psi4
import pennylane.numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)

mol = psi4.geometry("""
        1 1
        H    0.0056957528    0.0235477326    0.0000000000
        H    0.5224540730    0.8628715457    0.0000000000
        H    0.9909500019   -0.0043172515    0.0000000000
        symmetry c1
        """)

psi4.core.set_output_file('output.dat', False)

psi4.set_options({'basis': 'sto-3g',
                      'scf_type': 'pk',
                      'pcm': 'true',
                      'pcm_scf_type': 'separate',
                       'e_convergence': 1e-8,
                        'd_convergence': 1e-8,
               #         'freeze_core': True,
                        'print': 5})


psi4.driver.p4util.pcm_helper("""Units = Angstrom
      Medium {
         SolverType = IEFPCM
            Solvent = DMSO
               }

                  Cavity {
                     RadiiSet = UFF
                        Type = GePol
                           Scaling = False
                              Area = 0.3
                                 Mode = Implicit
                                    }
                                    """)

print('\nStarting SCF and integral build...')
t = time.time()

# First compute SCF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

vqe_options = {'max_iterations': 20,
               'PCM': True,
               'conv_tol': 10e-8,
               'mapping': 'jordan_wigner',
               'ansatz': 'custom',
             #  'device': 'ibmq_mumbai',
               'simulation': 'statevector'}

#molecule_options = {'structure': 'h3+.xyz',
#                     'charge': 1,
#                      'multiplicity': 1,
#                      'core_orbitals': 0,
#                      'active_electrons': 2}

#### PSI4 ###

vqe = PCM_VQE(wfn, vqe_options)
vqe.transform_integrals()
vqe.build_rdms_observable()
vqe.set_openfermion_molecular_hamiltonian_jordan_wigner()
vqe.set_up_ansatz_and_guess()
vqe.set_optimizer()

result = vqe.run()

#np.save("vqe_pcm_result.npy", result)
np.save("vqe_pcm_result_energy.npy", result["energy"])
np.save("vqe_pcm_result_polarization_energy.npy", result["polarization_energy"])
np.save("vqe_parameters.npy", result["params"])
np.save("vqe_estimated_electrons.npy", result["estimated_electrons"])
np.save("vqe_1rdm.npy", result["density_matrix"])
