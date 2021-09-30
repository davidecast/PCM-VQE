import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
from functools import partial
from pennylane.templates import UCCSD

geometry = 'h3+.xyz'

charge = 1

multiplicity = 1

basis_set = 'sto-3g'

symbols, coordinates = qchem.read_structure(geometry)

h, qubits = qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        charge = charge,
        package = 'psi4',
        mult = multiplicity,
        basis = basis_set,
        mapping = 'jordan_wigner'
        )

print(h)

dev = qml.device('default.qubit', wires=qubits)

electrons = 2

ref_state = qchem.hf_state(electrons, qubits)
singles, doubles = qchem.excitations(electrons, qubits)
s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)
dev = qml.device('default.qubit', wires=qubits)
circuit = partial(UCCSD, init_state=ref_state, s_wires=s_wires, d_wires=d_wires)
initial_guess = np.random.normal(0, np.pi, len(singles) + len(doubles))

cost_fn = qml.ExpvalCost(circuit, h, dev)

grad_h = qml.finite_diff(cost_fn)

opt = qml.GradientDescentOptimizer(stepsize=0.4)

params = initial_guess

result = {'energy': [], 'convergence': [], 'params': []}

for n in range(2):
    params, prev_energy = opt.step_and_cost(cost_fn, params, grad_fn = grad_h)
    energy = cost_fn(params)
    conv = np.abs(energy - prev_energy)
    result['energy'].append(energy)
    result['convergence'].append(conv)
    if n % 1 ==0:
        print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, energy))
    if n % 20 == 0:
        result['params'].append(params)
    if conv <= 1e-10:
        break
print('Final value of the ground-state energy = {:.8f} Ha'.format(energy))

#np.save("result_gas_phase.npy", result)

#np.save("energy.npy", result["energy"])
