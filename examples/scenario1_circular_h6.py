import tequila as tq
import numpy as np
import openfermion as of
import scipy
import os, sys

import mvb
sys.path.append(os.path.abspath("../measurement-bases/"))

import warnings
warnings.filterwarnings("ignore", category=tq.TequilaWarning)

geom = """H 0.0000 1.3970 0.0000
H 1.2098 0.6985 0.0000
H 1.2098 -0.6985 0.0000
H 0.0000 -1.3970 0.0000
H -1.2098 -0.6985 0.0000
H -1.2098 0.6985 0.0000"""
mol = tq.Molecule(geometry=geom, basis_set="sto-3g").use_native_orbitals()
fci = mol.compute_energy("fci")
print(f"fci: {fci}")
H = mol.make_hamiltonian()

graphs = []
graphs.append([(0,1),(2,3),(4,5)])
graphs.append([(1,2),(3,4),(0,5)])
graphs.append([(0,3),(1,2),(4,5)])
graphs.append([(2,5),(0,1),(3,4)])
graphs.append([(0,5),(1,4),(2,3)])

# Create true wave function and dummy circuit
Hof = H.to_openfermion()
Hsparse = of.linalg.get_sparse_operator(Hof)
v,vv = scipy.sparse.linalg.eigsh(Hsparse, k=10, ncv=200, sigma=fci)
# print(f"error: {fci - v[0]}") # check
wfn = tq.QubitWaveFunction.from_array(vv[:,0])
energy = wfn.inner(H * wfn).real
U = tq.gates.Rz(angle=0, target=range(H.n_qubits)) # dummy circuit

URs, variables, blocks = mvb.initialize_rotators(mol, graphs, solve_GNM=False, fix_gnm_values=True, add_relax=False, silent=False)
variables = {}
for block in blocks:
    variables.update({v:0 for v in block.extract_variables() if "r" in str(v)})

debug = False
if debug:
    print("------------------------------")
    print("TEST 1")
    # transform both H and U
    UR = blocks[0].get_UR_circuit().map_variables(variables)
    HX = mvb.fold_rotators(mol, UR).make_hamiltonian()
    EX = tq.ExpectationValue(H=HX, U=U+UR)
    print(f"transform both H and U: {tq.simulate(EX, initial_state=wfn)}")
    print(f"energy:          {energy}")
    print(energy - tq.simulate(EX, initial_state=wfn))
    print(np.isclose(energy, tq.simulate(EX, initial_state=wfn)))
    print("done")

    print("TEST 2")
    # sum of E1 and E2 is the same as E
    H1 = get_one_body_operator(mol)
    H2 = get_two_body_operator(mol)
    E1 = tq.ExpectationValue(U=U, H=H1)
    print(f"E1: {tq.simulate(E1, initial_state=wfn)}")
    E2 = tq.ExpectationValue(U=U, H=H2)
    print(energy - tq.simulate(E1 + E2, initial_state=wfn))
    print(f"result.energy == E1+E2: {np.isclose(energy,tq.simulate(E1 + E2, initial_state=wfn))}")
    print("done")

print("------------------------------")
print(f"Error in native basis")
# H2 = get_two_body_operator(mol)
E2 = tq.ExpectationValue(U=U, H=H)
hcb_mol,non_hcb_mol = mvb.get_hcb_part(mol)
EX = tq.ExpectationValue(U=U, H=hcb_mol.make_hamiltonian())
energy = tq.simulate(E2, initial_state=wfn)
hcb_two_body_part_1 = tq.simulate(EX, initial_state=wfn)
print(f"energy:    {tq.simulate(E2, initial_state=wfn)}")
print(f"hcb_two_body_part_1:   {hcb_two_body_part_1}")
print(f"error in native basis: {energy-hcb_two_body_part_1}")

print("------------------------------")
approx = 0.0
errors = []
# extract two body part
# _,h,g = mol.get_integrals()
# mol = tq.Molecule(geometry=mol.parameters.geometry, nuclear_repulsion=0.0, one_body_integrals=np.zeros(shape=[np.shape(h)[0],np.shape(h)[1]]),
#                   two_body_integrals=g, basis_set=mol.parameters.basis_set)

# fix rotators
spa_rotator = mvb.Block(UR=blocks[0].UR).map_variables(variables)
new_rotations = [
    mvb.Block(UR=blocks[1].UR).map_variables(variables),
    mvb.Block(UR=blocks[2].UR).map_variables(variables),
    mvb.Block(UR=blocks[3].UR).map_variables(variables),
    mvb.Block(UR=blocks[4].UR).map_variables(variables),
]

if debug:
    # transform both H and U
    print("TEST 3")
    test_mol = mvb.fold_rotators(mol, spa_rotator.get_UR_circuit())
    ETEST = tq.ExpectationValue(H=test_mol.make_hamiltonian(), U=U+spa_rotator.get_UR_circuit())
    print(f"energy:     {energy}")
    print(f"transform both H and U: {tq.simulate(ETEST, initial_state=wfn)}")
    print(f"test:                   {energy - tq.simulate(ETEST, initial_state=wfn)}")
    print(np.isclose(energy, tq.simulate(ETEST, initial_state=wfn)))
    print("done\n")

    # # measure H rotated in SPA with SPA circuit
    # print("TEST 4")
    # E_test = tq.ExpectationValue(U=blocks[0].construct_circuit(), H=mol.make_hamiltonian())
    # energy_test = tq.simulate(E_test, initial_state=wfn)
    # print(f"energy_test: {energy_test}")
    # tmol = mvb.fold_rotators(mol, blocks[0].get_UR_circuit().map_variables(variables1))
    # tcircuit = blocks[0].construct_circuit() + blocks[0].get_UR_circuit().map_variables(variables1)
    # hcb_mol, non_hcb_mol = mvb.get_hcb_part(tmol)
    # EX = tq.ExpectationValue(U=tcircuit, H=hcb_mol.make_hamiltonian())
    # incr = tq.simulate(EX, variables=variables1)
    # approx += incr
    # print(f"incr:               {incr}")
    # print(f"error in new basis: {energy_test-approx}")
    # print(f"new approx:         {approx}")
    # print("### 'error in new basis' should be zero ###")
    # print("done\n")

print(f"Remove SPA part")
approx = 0.0
tmol = mvb.fold_rotators(mol, spa_rotator.get_UR_circuit())
tcircuit = U + spa_rotator.get_UR_circuit()
if debug:
    # TESTING
    E1 = tq.ExpectationValue(H=mol.make_hamiltonian(), U=U)
    E2 = tq.ExpectationValue(H=tmol.make_hamiltonian(), U=tcircuit)
    test = E1 - E2
    print("test1 : ", tq.simulate(test, initial_state=wfn))
    # END
hcb_mol, non_hcb_mol = mvb.get_hcb_part(tmol)
if debug:
    # TESTING
    E3 = tq.ExpectationValue(H=tmol.make_hamiltonian(), U=U)
    E1 = tq.ExpectationValue(H=hcb_mol.make_hamiltonian(), U=U)
    E2 = tq.ExpectationValue(H=non_hcb_mol.make_hamiltonian(), U=U)
    test = E3 - (E1 + E2)
    print("test2 : ", tq.simulate(test, initial_state=wfn))
    # END
    # TESTING
    print(f"test3 (g): {np.allclose((mvb.fold_rotators(hcb_mol, spa_rotator.get_UR_circuit().dagger()).get_integrals()[2].elems + mvb.fold_rotators(non_hcb_mol, spa_rotator.get_UR_circuit().dagger()).get_integrals()[2].elems), mol.get_integrals()[2].elems)}")
    H1 = (hcb_mol.make_hamiltonian() + non_hcb_mol.make_hamiltonian()).simplify(1e-7)
    H2 = tmol.make_hamiltonian().simplify(1e-7)
    print(f"test3 (H): {all(p in H2.paulistrings for p in H1.paulistrings) and all(p in H1.paulistrings for p in H2.paulistrings) and np.allclose([H1[term] for term in H1.keys()], [H2[term] for term in H1.keys()], atol=1e-7)}")
    # END
EX = tq.ExpectationValue(U=tcircuit, H=hcb_mol.make_hamiltonian())
incr = tq.simulate(EX, initial_state=wfn)
approx += incr

M_tot = mvb.compute_num_meas(wfn, is_hcb=True, hcb_mol=hcb_mol, debug=False)
print(f"M_tot:              {M_tot:e}")
if debug:
    E_shots = tq.simulate(EX, initial_state=wfn, samples=int(M_tot), backend="qulacs")
    print(f"Energy with shots: {E_shots}")
    print(f"Error with shots: {(E_shots-incr):e}")
print(f"incr:               {incr}")
print(f"error in new basis: {energy-approx}")
mol = non_hcb_mol
errors.append(energy-approx)
print("------------------------------")

# Full procedure
target = energy - approx
approx = 0.0
old_UR = spa_rotator.get_UR_circuit()
for i,rotation in enumerate(new_rotations):
    _, mol, approx, _ = mvb.rotate_and_hcb(UR=rotation.get_UR_circuit(), circuit=U,
                                       molecule=mol, variables=variables, approx=approx, target=target,
                                       old_UR=old_UR, debug=debug, initial_state=wfn, true_wfn=wfn)
    old_UR = rotation.get_UR_circuit()
    errors.append(target-approx)
    print("------------------------------")

print(f"errors = {errors}")
