import tequila as tq
import numpy as np
import os, sys
import openfermion as of
import scipy
sys.path.append(os.path.abspath("../measurement-bases/"))
os.chdir(os.path.dirname(os.path.abspath(__file__))) # set working directory to file directory
from utils.block_utils import Block, BlockCircuit, gates_to_orb_rot
from utils.initializer_utils import initialize_rotators, initialize_blockcircuit, optimize, get_qpic, make_relax
from utils.qpic_visualization import OrbitalRotatorGate, PairCorrelatorGate, GenericGate
from utils.measurement_utils import fold_rotators, get_one_body_operator, get_two_body_operator, get_hcb_part, rotate_and_hcb, compute_num_meas

mol = tq.Molecule(geometry="H 1.5 0.0 0.0\nH 0.0 0.0 0.0\nH 1.5 0.0 1.5\nH 0.0 0.0 1.5", basis_set="sto-3g").use_native_orbitals()
fci = mol.compute_energy("fci")
print(f"fci: {fci}")
H = mol.make_hamiltonian()

Hof = mol.make_hamiltonian().to_openfermion()
Hsparse = of.linalg.get_sparse_operator(Hof)
v,vv = scipy.sparse.linalg.eigsh(Hsparse, sigma=mol.compute_energy("fci"))
# print(f"error: {fci - v[0]}") # check
wfn = tq.QubitWaveFunction.from_array(vv[:,0])
# print(f"wfn: {wfn}")

graphs = [
    [(0,1),(2,3)],
    [(0,3),(1,2)],
    [(0,2),(1,3)]
]
blocks = [
    # Block(UR=[OrbitalRotatorGate(0,1,"r01a"), OrbitalRotatorGate(2,3,"r23a")], UC=GenericGate(U=mol.make_ansatz("spa", edges=[(0,1),(2,3)]), name="initialstate", n_qubits_is_double=True), initial_state=True),
    Block(UR=[OrbitalRotatorGate(0,1,"r01a"), OrbitalRotatorGate(2,3,"r23a")], UC=[GenericGate(U=tq.gates.X([0,1,4,5]), name="initialstate"), PairCorrelatorGate(0,1,"c01a"), PairCorrelatorGate(2,3,"c23a")], initial_state=True),
    # Block(UR=[OrbitalRotatorGate(0,3,"r03"), OrbitalRotatorGate(1,2,"r12")], UC=[PairCorrelatorGate(0,3,"c03"), PairCorrelatorGate(1,2,"c12")]),
    Block(UR=[OrbitalRotatorGate(0,2,"r02"), OrbitalRotatorGate(1,3,"r13")], UC=[PairCorrelatorGate(0,2,"c02"), PairCorrelatorGate(1,3,"c13")]),
    Block(UR=[OrbitalRotatorGate(0,1,"r01b"), OrbitalRotatorGate(2,3,"r23b")], UC=[PairCorrelatorGate(0,1,"c01b"), PairCorrelatorGate(2,3,"c23b")])
]

blockcircuit = BlockCircuit(blocks)
init_vars = {v:np.pi/2 for v in blockcircuit.extract_variables() if "r" in str(v)}
# result = optimize(blockcircuit, H, fci, method="L-BFGS-B", initial_values=init_vars, silent=True, maxiter=200)
# for _ in range(100):
#     result = optimize(blockcircuit, H, fci, method="L-BFGS-B", initial_values="random", silent=False, maxiter=200)
# energy2 = result.energy
# variables2 = result.variables
energy2 = -1.9540486255440728
print("\nThree-graph")
print(f"error: {(energy2 - fci)*1000}")
variables2 = {"c01a": 9.795537127576827, "c23a": 10.221052049477567, "c02": 8.074927382534838, "c13": 1.7983902513867092, "c01b": 2.1874508602550917, "c23b": 7.920448582034537} # variables corresponding to error = 1mEh
variables2 = {**variables2, **init_vars}

# Create rotators
URs, variables, blocks = initialize_rotators(mol, graphs, solve_GNM=False, fix_gnm_values=True, add_relax=False, silent=True)
variables = {}
for block in blocks:
    variables.update({v:0 for v in block.extract_variables() if "r" in str(v)})

debug = True
if debug:
    print("------------------------------")
    print("TEST 1")
    # transform both H and U
    UR = blocks[0].get_UR_circuit().map_variables(variables)
    HX = fold_rotators(mol, UR).make_hamiltonian()
    EX = tq.ExpectationValue(H=HX, U=blockcircuit.construct_circuit()+UR)
    print(f"transform both H and U: {tq.simulate(EX, variables2)}")
    print(f"energy2:          {energy2}")
    print(energy2 - tq.simulate(EX, variables2))
    print(np.isclose(energy2, tq.simulate(EX, variables2)))
    print("done")

    print("TEST 2")
    # sum of E1 and E2 is the same as E
    H1 = get_one_body_operator(mol)
    H2 = get_two_body_operator(mol)
    E1 = tq.ExpectationValue(U=blockcircuit.construct_circuit(), H=H1)
    E2 = tq.ExpectationValue(U=blockcircuit.construct_circuit(), H=H2)
    print(energy2 - tq.simulate(E1 + E2, variables=variables2))
    print(f"energy2 == E1+E2: {np.isclose(energy2,tq.simulate(E1 + E2, variables=variables2))}")
    print("done")

print("------------------------------")
print(f"Error in native basis")
# H2 = get_two_body_operator(mol)
E2 = tq.ExpectationValue(U=blockcircuit.construct_circuit(), H=H)
hcb_mol,non_hcb_mol = get_hcb_part(mol)
EX = tq.ExpectationValue(U=blockcircuit.construct_circuit(), H=hcb_mol.make_hamiltonian())
energy2 = tq.simulate(E2, variables=variables2)
hcb_two_body_part_1 = tq.simulate(EX, variables=variables2)
print(f"energy2:    {tq.simulate(E2, variables=variables2)}")
print(f"hcb_two_body_part_1:   {hcb_two_body_part_1}")
print(f"error in native basis: {energy2-hcb_two_body_part_1}")

print("------------------------------")
approx = 0.0
errors = []
# extract two body part
# _,h,g = mol.get_integrals()
# mol = tq.Molecule(geometry=mol.parameters.geometry, nuclear_repulsion=0.0, one_body_integrals=np.zeros(shape=[np.shape(h)[0],np.shape(h)[1]]),
#                   two_body_integrals=g, basis_set=mol.parameters.basis_set)

# fix rotators
spa_rotator = Block(UR=blocks[0].UR).map_variables(variables)
new_rotations = [
    Block(UR=blocks[1].UR).map_variables(variables),
    Block(UR=blocks[2].UR).map_variables(variables)
]

if debug:
    # transform both H and U
    print("TEST 3")
    test_mol = fold_rotators(mol, spa_rotator.get_UR_circuit())
    ETEST = tq.ExpectationValue(H=test_mol.make_hamiltonian(), U=blockcircuit.construct_circuit()+spa_rotator.get_UR_circuit())
    print(f"energy2:     {energy2}")
    print(f"transform both H and U: {tq.simulate(ETEST, variables2)}")
    print(f"test:                   {energy2 - tq.simulate(ETEST, variables2)}")
    print(np.isclose(energy2, tq.simulate(ETEST, variables2)))
    print("done\n")

print(f"Remove SPA part")
approx = 0.0
tmol = fold_rotators(mol, spa_rotator.get_UR_circuit())
tcircuit = blockcircuit.construct_circuit() + spa_rotator.get_UR_circuit()
if debug:
    # TESTING
    E1 = tq.ExpectationValue(H=mol.make_hamiltonian(), U=blockcircuit.construct_circuit())
    E2 = tq.ExpectationValue(H=tmol.make_hamiltonian(), U=tcircuit)
    test = E1 - E2
    print("test1 : ", tq.simulate(test, variables2))
    # END
hcb_mol, non_hcb_mol = get_hcb_part(tmol)
if debug:
    # TESTING
    E3 = tq.ExpectationValue(H=tmol.make_hamiltonian(), U=blockcircuit.construct_circuit())
    E1 = tq.ExpectationValue(H=hcb_mol.make_hamiltonian(), U=blockcircuit.construct_circuit())
    E2 = tq.ExpectationValue(H=non_hcb_mol.make_hamiltonian(), U=blockcircuit.construct_circuit())
    test = E3 - (E1 + E2)
    print("test2 : ", tq.simulate(test, variables2))
    # END
    # TESTING
    print(f"test3 (g): {np.allclose((fold_rotators(hcb_mol, spa_rotator.get_UR_circuit().dagger()).get_integrals()[2].elems + fold_rotators(non_hcb_mol, spa_rotator.get_UR_circuit().dagger()).get_integrals()[2].elems), mol.get_integrals()[2].elems)}")
    H1 = (hcb_mol.make_hamiltonian() + non_hcb_mol.make_hamiltonian()).simplify(1e-7)
    H2 = tmol.make_hamiltonian().simplify(1e-7)
    print(f"test3 (H): {all(p in H2.paulistrings for p in H1.paulistrings) and all(p in H1.paulistrings for p in H2.paulistrings) and np.allclose([H1[term] for term in H1.keys()], [H2[term] for term in H1.keys()], atol=1e-7)}")
    # END
EX = tq.ExpectationValue(U=tcircuit, H=hcb_mol.make_hamiltonian())
incr = tq.simulate(EX, variables=variables2)
EY = tq.ExpectationValue(U=tcircuit, H=non_hcb_mol.make_hamiltonian())
rest = tq.simulate(EY, variables=variables2)
approx += incr
M_tot = compute_num_meas(wfn, is_hcb=True, hcb_mol=hcb_mol, debug=False)
print(f"M_tot:              {M_tot:e}")
if debug:
    print(f"test:               {energy2-approx-rest}")
    E_shots = tq.simulate(EX, variables=variables2, samples=int(M_tot), backend="qulacs")
    print(f"Energy with shots: {E_shots}")
    print(f"Error with shots: {(E_shots-incr):e}")
print(f"incr:               {incr}")
print(f"error in new basis: {energy2-approx}")
mol = non_hcb_mol
errors.append(energy2-approx)
print("------------------------------")

# Full procedure
target = energy2 - approx
approx = 0.0
old_UR = spa_rotator.get_UR_circuit()
for i,rotation in enumerate(new_rotations):
    _, mol, approx, _ = rotate_and_hcb(UR=rotation.get_UR_circuit(), circuit=blockcircuit.construct_circuit(),
                                       molecule=mol, variables=variables2, approx=approx, target=target,
                                       old_UR=old_UR, debug=debug)
    old_UR = rotation.get_UR_circuit()
    errors.append(target-approx)
    print("------------------------------")

print(f"errors = {errors}")