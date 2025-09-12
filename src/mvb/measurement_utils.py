import tequila as tq
import numpy as np
import openfermion as of
import scipy
import time
import copy

from .block_utils import gates_to_orb_rot

def fold_rotators(mol, UR):
    # Get the transformation matrix from the circuit
    UR_matrix = gates_to_orb_rot(UR, mol.n_orbitals)

    # Rotate one- and two-body part
    c,h,g = mol.get_integrals()
    g = g.elems

    th = np.einsum("ix, jx -> ij",  h, UR_matrix, optimize='greedy')
    th = np.einsum("xj, ix -> ij", th, UR_matrix, optimize='greedy')
    # same as th = (UR_matrix.dot(h)).dot(UR_matrix.T)

    tg = np.einsum("ijkx, lx -> ijkl",  g, UR_matrix, optimize='greedy')
    tg = np.einsum("ijxl, kx -> ijkl", tg, UR_matrix, optimize='greedy')
    tg = np.einsum("ixkl, jx -> ijkl", tg, UR_matrix, optimize='greedy')
    tg = np.einsum("xjkl, ix -> ijkl", tg, UR_matrix, optimize='greedy')

    tmol = tq.Molecule(parameters=mol.parameters, transformation=mol.transformation, n_electrons=mol.n_electrons,
                       nuclear_repulsion=c, one_body_integrals=th, two_body_integrals=tg, basis_set=mol.parameters.basis_set,
                       ordering='openfermion')

    return tmol

def get_one_body_operator(mol):
    c,h,g = mol.get_integrals()
    n = h.shape[0]
    dummy = tq.Molecule(parameters=mol.parameters, transformation=mol.transformation, n_electrons=mol.n_electrons,
                        one_body_integrals=h, two_body_integrals=np.zeros([n,n,n,n]), nuclear_repulsion=c,
                        ordering='openfermion')
    return dummy.make_hamiltonian()
def get_two_body_operator(mol):
    c,h,g = mol.get_integrals()
    n = h.shape[0]
    dummy = tq.Molecule(parameters=mol.parameters, transformation=mol.transformation, n_electrons=mol.n_electrons,
                        one_body_integrals=np.zeros([n,n]), two_body_integrals=g, nuclear_repulsion=0.0,
                        ordering='openfermion')
    return dummy.make_hamiltonian()

def get_hcb_part(mol, diagonal=False):
    c,h,g = mol.get_integrals()

    hcb_h = np.zeros(shape=(h.shape[0], h.shape[1]))
    non_hcb_h = np.zeros(shape=(h.shape[0], h.shape[1]))
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            if i==j:
                hcb_h[i][j] = h[i][j] # ii
            else:
                non_hcb_h[i][j] = h[i][j]

    hcb_g = np.zeros(shape=(g.shape[0], g.shape[1], g.shape[2], g.shape[3]))
    non_hcb_g = np.zeros(shape=(g.shape[0], g.shape[1], g.shape[2], g.shape[3]))
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            for k in range(g.shape[2]):
                for l in range(g.shape[3]):
                    if diagonal:
                        # following the ordering i,l = spin-up and j,k = spin-down
                        if (i==l and j==k): # ikki
                            hcb_g[i][j][k][l] = g.elems[i][j][k][l]
                        else:
                            non_hcb_g[i][j][k][l] = g.elems[i][j][k][l]
                    else:
                        if (i==j and k==l) or (i==l and j==k) or (i==k and j==l): # iikk or ikki or ijij
                            hcb_g[i][j][k][l] = g.elems[i][j][k][l]
                        else:
                            non_hcb_g[i][j][k][l] = g.elems[i][j][k][l]

    # With c and h
    hcb_mol = tq.Molecule(parameters=mol.parameters, transformation=mol.transformation, n_electrons=mol.n_electrons,
                          nuclear_repulsion=c, one_body_integrals=hcb_h, two_body_integrals=hcb_g, ordering='openfermion')
    non_hcb_mol = tq.Molecule(parameters=mol.parameters, transformation=mol.transformation, n_electrons=mol.n_electrons,
                              nuclear_repulsion=0, one_body_integrals=non_hcb_h, two_body_integrals=non_hcb_g, ordering='openfermion')

    return hcb_mol, non_hcb_mol

def sum_two_body_parts(molecule1, molecule2):
    c1,h1,g1 = molecule1.get_integrals()
    c2,h2,g2 = molecule2.get_integrals()

    sum_g = g1.elems + g2.elems

    sum_mol = tq.Molecule(parameters=molecule1.parameters, transformation=molecule1.transformation, n_electrons=molecule1.n_electrons,
                          nuclear_repulsion=0.0, one_body_integrals=np.zeros(shape=[np.shape(h1)[0],np.shape(h1)[1]]),
                          two_body_integrals=sum_g, odering='openfermion')
    
    return sum_mol

def rotate_and_hcb(UR, circuit, molecule, variables, approx, target, old_UR, diagonal=False, debug=False, silent=False, *args, **kwargs):

    # Rotate the two body part
    tmol = fold_rotators(molecule, old_UR.dagger()+UR)
    tcircuit = circuit + UR

    if debug:
        # TESTING
        E1 = tq.ExpectationValue(H=molecule.make_hamiltonian(), U=circuit+old_UR)
        E2 = tq.ExpectationValue(H=tmol.make_hamiltonian(), U=tcircuit)
        test = E1 - E2
        print("test1 : ", tq.simulate(test, variables))
        # END
    
    # Extract the hcb part and the non-hcb part of the transformed molecule
    hcb_mol, non_hcb_mol = get_hcb_part(tmol, diagonal=diagonal)

    if debug:
        # TESTING
        E3 = tq.ExpectationValue(H=tmol.make_hamiltonian(), U=tcircuit)
        E1 = tq.ExpectationValue(H=hcb_mol.make_hamiltonian(), U=tcircuit)
        E2 = tq.ExpectationValue(H=non_hcb_mol.make_hamiltonian(), U=tcircuit)
        test = E3 - (E1 + E2)
        print("test2 : ", tq.simulate(test, variables))
        # END
    
    if debug:
        # Test two-body part and Hamiltonian consistency
        print(f"test3 (g): {np.allclose((fold_rotators(hcb_mol, UR.dagger()+old_UR).get_integrals()[2].elems + fold_rotators(non_hcb_mol, UR.dagger()+old_UR).get_integrals()[2].elems), molecule.get_integrals()[2].elems)}")
        H1 = (hcb_mol.make_hamiltonian() + non_hcb_mol.make_hamiltonian()).simplify(1e-7)
        H2 = tmol.make_hamiltonian().simplify(1e-7)
        print(f"test3 (H): {all(p in H2.paulistrings for p in H1.paulistrings) and all(p in H1.paulistrings for p in H2.paulistrings) and np.allclose([H1[term] for term in H1.keys()], [H2[term] for term in H1.keys()], atol=1e-7)}")

    # Compute expectation values and error
    EX = tq.ExpectationValue(U=tcircuit, H=hcb_mol.make_hamiltonian())
    incr = tq.simulate(EX, variables=variables, *args, **kwargs)
    EY = tq.ExpectationValue(U=tcircuit, H=non_hcb_mol.make_hamiltonian())
    rest = tq.simulate(EY, variables=variables, *args, **kwargs)
    approx += incr # approx = approximation of the target
    # norm = np.linalg.norm(non_hcb_mol.make_hamiltonian().to_matrix())

    if not silent:
        if 'true_wfn' in kwargs:
            wfn = kwargs['true_wfn']
        else:
            wfn = tq.simulate(circuit, variables)
        M_tot = compute_num_meas(wfn, is_hcb=True, hcb_mol=hcb_mol, debug=False, *args, **kwargs)
        print(f"M_tot:              {M_tot:e}")
        # if debug:
        #     E_samples = tq.simulate(EX, variables=variables, samples=int(M_tot), backend="qulacs")
        #     print(f"Energy with samples: {E_samples}")
        #     print(f"Error with samples: {(E_samples-incr):e}")
        print(f"incr:               {incr}")
        # print(f"rest:               {rest}")
        print(f"error in new basis: {target-approx}")
        # print(f"operator norm:      {norm}")
        if debug:
            print(f"test:               {target-approx-rest}")
        print(f"new approx:         {approx}")

        return hcb_mol, non_hcb_mol, approx, M_tot

    return hcb_mol, non_hcb_mol, approx


def compute_num_meas(wfn, groups=None, transformations=None, is_hcb=False, hcb_mol=None, eps=1e-3, remove_Z=False, n_repetitions=1, debug=False, *args, **kwargs):
    U = tq.gates.Rz(angle=0, target=range(wfn.n_qubits)) # dummy empty circuit
    
    # If hcb split in the three groups
    if is_hcb:
        H1 = tq.QubitHamiltonian()
        H2 = tq.QubitHamiltonian()
        H3 = tq.QubitHamiltonian()
        for p in hcb_mol.make_hamiltonian().paulistrings:
            q = p.naked().qubits
            if p.is_all_z():
                H1 += tq.QubitHamiltonian().from_paulistrings(p)
            else:
                if (p.naked()[q[0]] == "X" and p.naked()[q[1]] == "X") or (p.naked()[q[0]] == "Y" and p.naked()[q[1]] == "Y"):
                    H2 += tq.QubitHamiltonian().from_paulistrings(p)
                else:
                    H3 += tq.QubitHamiltonian().from_paulistrings(p)
        groups = [H1, H2, H3]

    # Remove single-Z paulis from all groups
    if remove_Z:
        old_groups = copy.deepcopy(groups)
        if transformations is not None:
            old_transformations = copy.deepcopy(transformations)
        else:
            old_transformations = [tq.QCircuit() for k in range(len(groups))]
        for i, group in enumerate(groups):
            lst = group.paulistrings
            for op in group.paulistrings:
                if len(op)==1 and op.is_all_z():
                    lst.remove(op)
            groups[i] = tq.QubitHamiltonian.from_paulistrings(lst)

    # Compute group number of measurements
    M_tot = 0
    E_tot_test = 0
    E_tot_samples = 0
    for i, group in enumerate(groups):

        if is_hcb:
            # Diagonalize each of the three hcb groups
            fc_groups_and_unitaries, _ = tq.grouping.compile_groups.compile_commuting_parts(group, unitary_circuit='improved')
            group = fc_groups_and_unitaries[0][0]
            transformation = fc_groups_and_unitaries[0][1]
        else:
            transformation = transformations[i]

        # Compute individual number of measurements
        M_group = 1
        E_group_test = 0
        for op in group.paulistrings:
            E_group_test += op.coeff * tq.simulate(tq.ExpectationValue(U+transformation, tq.QubitHamiltonian().from_paulistrings(op.naked())), initial_state=wfn)
            E_tot_test += op.coeff * tq.simulate(tq.ExpectationValue(U+transformation, tq.QubitHamiltonian().from_paulistrings(op.naked())), initial_state=wfn)
            O1 = 1.0
            O2 = tq.simulate(tq.ExpectationValue(U+transformation, tq.QubitHamiltonian().from_paulistrings(op.naked())), initial_state=wfn) ** 2
            var = O1 - O2
            M_l = ((abs(op.coeff) * np.sqrt(var)) / eps) ** 2
            # print(f"op: {op}")
            # print(f"var: {var}")
            # print(f"exp_val: {op.coeff * tq.simulate(tq.ExpectationValue(U+transformation, tq.QubitHamiltonian().from_paulistrings(op.naked())), initial_state=wfn)}")
            
            # The number of measurements for each group is only the largest number of measurements in the group
            # print(f"M_l: {M_l:e}")
            # print(f"M_group: {M_group:e}")
            # print()
            if M_l>M_group:
                M_group = M_l

        if debug:
            E_group_true = tq.simulate(tq.ExpectationValue(U+transformation, group), initial_state=wfn)
            if remove_Z:
                # print(E_group_true)
                # print(tq.simulate(tq.ExpectationValue(U+old_transformations[i], old_groups[i]), initial_state=wfn))
                print(f"Remove-Z error group{i}: {E_group_true - tq.simulate(tq.ExpectationValue(U+old_transformations[i], old_groups[i]), initial_state=wfn):e}")

            # print(f"Error group {i}: {E_group_test - E_group_true:e}") # Measuring the full group at once is equal to measuring the single operators
            print(f"M_group{i}: {(M_group):e}")
            lst_E_samples = []
            print(f"n_repetitions: {n_repetitions}")
            for _ in range(n_repetitions):
                lst_E_samples.append(tq.simulate(tq.ExpectationValue(U+transformation, group), initial_state=wfn, samples=int(M_group), backend="qulacs"))
            E_tot_samples += (np.mean(lst_E_samples) - E_group_true)
            print(f"Error group{i} samples: {(np.mean(lst_E_samples) - E_group_true):e}") # We expect the error to be close to the precision or lower
            # print(f"Variance group{i} samples: {(sum([(abs(en - E_group_true))**2 for en in lst_E_samples]) / n_repetitions):e}")
            print()
        
        M_tot += M_group

    if debug:
        if remove_Z:
            # Construct H as the sum of groups
            new_H = sum(groups)
        elif is_hcb:
            new_H = hcb_mol.make_hamiltonian()
        else:
            new_H = kwargs["molecule"].make_hamiltonian()

        E_tot_true = tq.simulate(tq.ExpectationValue(U, new_H), initial_state=wfn)
        print(f"Error groups: {(E_tot_test - E_tot_true):e}") # Measuring the full observable is equal to measuring the single operators
        print(f"E_tot_samples: {E_tot_samples:e}")

        if remove_Z:
            if is_hcb:
                print(f"Remove-Z error: {E_tot_test - tq.simulate(tq.ExpectationValue(U, hcb_mol.make_hamiltonian()), initial_state=wfn):e}")    
            else:
                print(f"Remove-Z error: {E_tot_test - tq.simulate(tq.ExpectationValue(U, kwargs['molecule'].make_hamiltonian()), initial_state=wfn):e}")

    return M_tot
