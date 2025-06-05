# gin H6
import tequila as tq
from tequila.grouping.compile_groups import compile_commuting_parts 
import tequila as tq
from tequila.hamiltonian import QubitHamiltonian, PauliString
from tequila.grouping.binary_rep import BinaryPauliString, BinaryHamiltonian
from tequila.grouping.fermionic_functions import n_elec
from tequila.grouping.fermionic_methods import get_wavefunction
import numpy as np
import pandas as pd
import openfermion as of
import scipy

from mvb import compute_num_meas

df = pd.read_csv("h6_100")
df = df.iloc[:,10:]

n_molecules = []

for k in range(100):
    measurements = {}
    print(f"Molecule n{k}")

    # Define geometry
    geometry = f"""
H {df["x_0"][k]} {df["y_0"][k]} {df["z_0"][k]}
H {df["x_1"][k]} {df["y_1"][k]} {df["z_1"][k]}
H {df["x_2"][k]} {df["y_2"][k]} {df["z_2"][k]}
H {df["x_3"][k]} {df["y_3"][k]} {df["z_3"][k]}
H {df["x_4"][k]} {df["y_4"][k]} {df["z_4"][k]}
H {df["x_5"][k]} {df["y_5"][k]} {df["z_5"][k]}
"""

    mol = tq.Molecule(geometry=geometry, basis_set="sto-3g").use_native_orbitals()
    H = mol.make_hamiltonian()
    print(f"len(H): {len(H)}")
    measurements['original'] = len(H)

    Hof = H.to_openfermion()
    Hsparse = of.linalg.get_sparse_operator(Hof)
    v,vv = scipy.sparse.linalg.eigsh(Hsparse, sigma=mol.compute_energy("fci"))
    # print(f"error: {mol.compute_energy('fci') - v[0]}") # check
    wfn = tq.QubitWaveFunction.from_array(vv[:,0])
    
    ######## LF ########
    try:
        options = {
            'method': 'lf', # lf, rlf, si, ics, lr, fff-lr
            'condition': 'fc', # qwc, fc
            'optimize': 'yes',
            'n_el': mol.n_electrons
        }
        fc_groups_and_unitaries, sample_ratios = compile_commuting_parts(H, unitary_circuit='improved', options=options)

        number_fc_groups = len(fc_groups_and_unitaries)
        print(f'method: {options["method"]}, condition: {options["condition"]}')
        print('The number of groups to measure is: ', number_fc_groups)
        E = tq.ExpectationValue(H=H, U=tq.QCircuit(), optimize_measurements=options)
        md = 0
        for e in E.get_expectationvalues():
            depth = tq.compile_circuit(e.U).depth
            # print(depth)
            if depth > md: md = depth
        print("depth overhead: ", md)

        groups = []
        transformations = []
        for group, circuit in fc_groups_and_unitaries:
            groups.append(group)
            transformations.append(circuit)
        M_tot = compute_num_meas(wfn, groups, transformations, molecule=mol)
        print(f"M_tot: {M_tot:e}")
        measurements[options["method"]] = [number_fc_groups, md, M_tot]
    except:
        pass
    
    ######## RLF ########
    try:
        options = {
            'method': 'rlf', # lf, rlf, si, ics, lr, fff-lr
            'condition': 'fc', # qwc, fc
            'optimize': 'yes',
            'n_el': mol.n_electrons
        }
        fc_groups_and_unitaries, sample_ratios = compile_commuting_parts(H, unitary_circuit='improved', options=options)

        number_fc_groups = len(fc_groups_and_unitaries)
        print(f'method: {options["method"]}, condition: {options["condition"]}')
        print('The number of groups to measure is: ', number_fc_groups)
        E = tq.ExpectationValue(H=H, U=tq.QCircuit(), optimize_measurements=options)
        md = 0
        for e in E.get_expectationvalues():
            depth = tq.compile_circuit(e.U).depth
            # print(depth)
            if depth > md: md = depth
        print("depth overhead: ", md)

        groups = []
        transformations = []
        for group, circuit in fc_groups_and_unitaries:
            groups.append(group)
            transformations.append(circuit)
        M_tot = compute_num_meas(wfn, groups, transformations, molecule=mol)
        print(f"M_tot: {M_tot:e}")
        measurements[options['method']] = [number_fc_groups, md, M_tot]
    except:
        pass

    ######## SI ########
    try:
        options = {
            'method': 'si', # lf, rlf, si, ics, lr, fff-lr
            'condition': 'fc', # qwc, fc
            'optimize': 'yes',
            'n_el': mol.n_electrons
        }
        fc_groups_and_unitaries, sample_ratios = compile_commuting_parts(H, unitary_circuit='improved', options=options)
        
        number_fc_groups = len(fc_groups_and_unitaries)
        print(f'method: {options["method"]}, condition: {options["condition"]}')
        print('The number of groups to measure is: ', number_fc_groups)
        E = tq.ExpectationValue(H=H, U=tq.QCircuit(), optimize_measurements=options)
        md = 0
        for e in E.get_expectationvalues():
            depth = tq.compile_circuit(e.U).depth
            # print(depth)
            if depth > md: md = depth
        print("depth overhead: ", md)

        groups = []
        transformations = []
        for group, circuit in fc_groups_and_unitaries:
            groups.append(group)
            transformations.append(circuit)
        M_tot = compute_num_meas(wfn, groups, transformations, molecule=mol)
        print(f"M_tot: {M_tot:e}")
        measurements[options['method']] = [number_fc_groups, md, M_tot]
    except:
        pass

    # ######## ICS ########
    # try:
    #     Hbin = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    #     _, psis_appr = get_wavefunction(H.to_openfermion(), "cisd", f"h{mol.n_electrons}", mol.n_electrons, save=False)
    #     # print(mol, H, Hbin, psis_appr[0], len(Hbin.binary_terms) - 1)
    #     n_paulis = len(Hbin.binary_terms) - 1
    #     psi_appr = psis_appr[0]

    #     def prepare_cov_dict(H, psi_appr):
    #         '''
    #         Return the covariance dictionary containing Cov(P1, P2). 
    #         In a practical calculation, this covariance dictionary would be built from
    #         a Hartree-Fock or configuration interaction singles and doulbes (CISD) 
    #         wavefunction. Here, we use the CISD wavefunction.
    #         '''
    #         terms = H.binary_terms
    #         cov_dict = {}
    #         wfn0 = tq.QubitWaveFunction(psi_appr)
    #         for idx, term1 in enumerate(terms):
    #             for term2 in terms[idx:]:
    #                 pw1 = BinaryPauliString(term1.get_binary(), 1.0)
    #                 pw2 = BinaryPauliString(term2.get_binary(), 1.0)
    #                 op1 = QubitHamiltonian.from_paulistrings(pw1.to_pauli_strings())
    #                 op2 = QubitHamiltonian.from_paulistrings(pw2.to_pauli_strings())
    #                 if pw1.commute(pw2):
    #                     prod_op = op1 * op2
    #                     cov_dict[(term1.binary_tuple(), term2.binary_tuple())] = wfn0.inner(prod_op(wfn0)) - wfn0.inner(op1(wfn0)) * wfn0.inner(op2(wfn0))
    #         return cov_dict

    #     # mol, H, Hbin, psi_appr, n_paulis = prepare_test_hamiltonian()
    #     print("Number of Pauli products to measure: {}".format(n_paulis))
    #     cov_dict = prepare_cov_dict(Hbin, psi_appr)

    #     options = {
    #         'method': 'ics', # lf, rlf, si, ics, lr, fff-lr
    #         'condition': 'fc', # qwc, fc
    #         'optimize': 'yes',
    #         'n_el': mol.n_electrons,
    #         'cov_dict': cov_dict
    #     }
    #     fc_groups_and_unitaries, sample_ratios = compile_commuting_parts(H, unitary_circuit='improved', options=options)
        
    #     number_fc_groups = len(fc_groups_and_unitaries)
    #     print(f'method: {options["method"]}, condition: {options["condition"]}')
    #     print('The number of groups to measure is: ', number_fc_groups)
    #     E = tq.ExpectationValue(H=H, U=tq.QCircuit(), optimize_measurements=options)
    #     md = 0
    #     for e in E.get_expectationvalues():
    #         depth = tq.compile_circuit(e.U).depth
    #         # print(depth)
    #         if depth > md: md = depth
    #     print("depth overhead: ", md)

    #     groups = []
    #     transformations = []
    #     for group, circuit in fc_groups_and_unitaries:
    #         groups.append(group)
    #         transformations.append(circuit)
    #     M_tot = compute_num_meas(wfn, groups, transformations, molecule=mol)
    #     print(f"M_tot: {M_tot:e}")
    #     measurements[options['method']] = [number_fc_groups, md, M_tot]
    # except:
    #     pass

    ######## LR ########
    try:
        options = {
            'method': 'lr', # lf, rlf, si, ics, lr, fff-lr
            'condition': 'fc', # qwc, fc
            'optimize': 'yes',
            'n_el': mol.n_electrons
        }
        fc_groups_and_unitaries, sample_ratios = compile_commuting_parts(H, unitary_circuit='improved', options=options)

        number_fc_groups = len(fc_groups_and_unitaries)
        print(f'method: {options["method"]}, condition: {options["condition"]}')
        print('The number of groups to measure is: ', number_fc_groups)
        E = tq.ExpectationValue(H=H, U=tq.QCircuit(), optimize_measurements=options)
        md = 0
        for e in E.get_expectationvalues():
            depth = tq.compile_circuit(e.U).depth
            # print(depth)
            if depth > md: md = depth
        print("depth overhead: ", md)

        groups = []
        transformations = []
        for group, circuit in fc_groups_and_unitaries:
            groups.append(group)
            transformations.append(circuit)
        M_tot = compute_num_meas(wfn, groups, transformations, molecule=mol)
        print(f"M_tot: {M_tot:e}")
        measurements[options['method']] = [number_fc_groups, md, M_tot]
    except:
        pass

    ######## FFF-LR ########
    try:
        options = {
            'method': 'fff-lr', # lf, rlf, si, ics, lr, fff-lr
            'condition': 'fc', # qwc, fc
            'optimize': 'yes',
            'n_el': mol.n_electrons
        }
        fc_groups_and_unitaries, sample_ratios = compile_commuting_parts(H, unitary_circuit='improved', options=options)

        number_fc_groups = len(fc_groups_and_unitaries)
        print(f'method: {options["method"]}, condition: {options["condition"]}')
        print('The number of groups to measure is: ', number_fc_groups)
        E = tq.ExpectationValue(H=H, U=tq.QCircuit(), optimize_measurements=options)
        md = 0
        for e in E.get_expectationvalues():
            depth = tq.compile_circuit(e.U).depth
            # print(depth)
            if depth > md: md = depth
        print("depth overhead: ", md)

        groups = []
        transformations = []
        for group, circuit in fc_groups_and_unitaries:
            groups.append(group)
            transformations.append(circuit)
        M_tot = compute_num_meas(wfn, groups, transformations, molecule=mol)
        print(f"M_tot: {M_tot:e}")
        measurements[options['method']] = [number_fc_groups, md, M_tot]
    except:
        pass

    n_molecules.append(measurements)
    print(n_molecules)
    print()

print(n_molecules)
