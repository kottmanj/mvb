import tequila as tq
import numpy as np
import time
from .qpic_visualization import OrbitalRotatorGate, PairCorrelatorGate, GenericGate, export_to_qpic, qpic_to_png, qpic_to_pdf
from .block_utils import Block, BlockCircuit
from .qvb_utils import GNM
import os
import copy

def initialize_rotators(molecule, graphs, variables=None, custom_blocks=None, solve_GNM=True, fix_gnm_values=False, add_relax=False, relax_name="butterfly", custom_relax = None, static_rotators=False, pre_opt=True, silent=False, draw_circuits=False, *args, **kwargs):
    blocks = []
    URs = []
    if not variables:
        variables = {}

    # Create circuits or pass custom ones
    if custom_blocks:
        blocks = custom_blocks
    else:
        # Create SPA + UR.dagger() circuit for each graph
        for i,graph in enumerate(graphs):
            SPA = molecule.make_ansatz(name="SPA", edges=graph, label=i)
            SPA = GenericGate(SPA, name="initialstate", n_qubits_is_double=True)
            UR = []
            for edge in graph:
                UR.append(OrbitalRotatorGate(i=edge[0], j=edge[1], angle=(tq.Variable((f"main_r{edge}", i))+0.5)*np.pi, molecule=molecule, *args, **kwargs))
            blocks.append(Block(UR=UR, UC=SPA, initial_state=True))

            # Define and add relaxation
            if add_relax:
                if relax_name=="custom":
                    relaxation = custom_relax[i]
                else:
                    relaxation = make_relax(i, molecule, name=relax_name)
                    if i==0 and (relax_name=="butterfly" or relax_name=="1234"):
                        relaxation = relaxation[3:] # remove overlapping relax for first block
                blocks[i].add_UR(relaxation, overwrite=True)

    # Create the list of rotator gates (URs)
    for i,block in enumerate(blocks):
        URs.append(blocks[i].UR)

    # Show the circuits (called blocks) used as basis
    if not silent:
        for i,block in enumerate(blocks):
            print(f"block{i}: {block}")

    # Draw circuits
    if draw_circuits:
        for i,block in enumerate(blocks):
            export_to_qpic(block.construct_visual(), filename=f"block_visual{i}", filepath=os.getcwd())
            # block.construct_circuit().export_to(filename=f"{os.getcwd()}/block{i}.qpic")

    # Solve GNM
    N = len(blocks)
    if solve_GNM:
        if not fix_gnm_values:
            if pre_opt:
                for i in range(N):
                    print(f"Pre-optimization circuit: {i}")
                    E = tq.ExpectationValue(H=molecule.make_hamiltonian(), U=blocks[i].construct_circuit())
                    result = tq.minimize(E, silent=True)
                    variables.update(result.variables)
                    print(f"error: {(result.energy - molecule.compute_energy('fci'))*1000}")

            constr_blocks = [circ.construct_circuit() for circ in blocks]
            print(f"\nStarting GNM calculation, N={N}, M={N}:")
            start_time = time.time()
            v,vv,variables = GNM(circuits=constr_blocks, M=N, variables=variables, H=molecule.make_hamiltonian(), silent=True)
            tot_time = time.time() - start_time
        else:
            # set precalculated values (H6 only)
            print(f"Skip GNM calculation, N={N}, M={N}:")
            v, tot_time, variables = fix_gnm(length=N, add_relax=add_relax, relax_name=relax_name)

        # Print results
        error = v[0] - molecule.compute_energy('fci')
        print(f"error: {error*1000}")
        print(f"time: {tot_time}s, {tot_time/60}m, {tot_time/60/60}h")
        depths = []
        for block in blocks:
            depths.append(tq.compile_circuit(block.construct_circuit()).depth)
        if not silent:
            print(f"energy: {v[0]}")
            print(f"variables: {variables}")
            print(f"num_variables: {len(variables)}")
            print(f"depths: {depths}")
        
        # Fix rotators
        if static_rotators:
            for UR in URs:
                for rot in UR:
                    rot.angle = rot.angle.map_variables(variables)
        return URs, variables, blocks, v

    else:
        for block in blocks:
            variables.update({v:0.0 for v in block.extract_variables()})

    return URs, variables, blocks


def initialize_blockcircuit(molecule, graphs, URs=None, input_blocks=None, ordering=[0,1,2,3], draw_circuits=False):
    N = len(graphs)
    
    if input_blocks and not URs:
        # Create URs
        URs = []
        for i,graph in enumerate(graphs):
            URs.append(input_blocks[i].UR)

    # Create blocks
    blocks = []
    SPA = molecule.make_ansatz(name="SPA", edges=graphs[ordering[0]], label=ordering[0])
    SPA = GenericGate(SPA, name="initialstate", n_qubits_is_double=True)
    blocks.append(Block(UR=URs[0],UC=SPA,initial_state=True))
    if N>1:
        for i in ordering[1:N]:
            UC = []
            for edge in graphs[i]:
                UC.append(PairCorrelatorGate(i=edge[0], j=edge[1], angle=(edge, "c", i)))
            blocks.append(Block(UR=URs[i],UC=UC))

    # Create blockcircuit
    blockcircuit = BlockCircuit()
    for j in ordering[:len(graphs)]:
        blockcircuit += blocks[j]

    # Draw blockcircuit
    if draw_circuits:
        print(f"\nDrawing blockcircuit: {blockcircuit}")
        export_to_qpic(blockcircuit.construct_visual(), filename="blockcircuit_visual", filepath=os.getcwd())
        # blockcircuit.construct_circuit().export_to(filename=f"{os.getcwd()}/blockcircuit.qpic")

    return blockcircuit


def optimize(circuit, H, fci=None, initial_values=None, silent=False, *args, **kwargs):
    if hasattr(circuit, "construct_circuit"):
        E = tq.ExpectationValue(H=H, U=circuit.construct_circuit())
    else:
        E = tq.ExpectationValue(H=H, U=circuit)
    zero_vars = {v:0 for v in E.extract_variables()}
    if not initial_values:
        initial_values = zero_vars
    elif initial_values != "random":
        initial_values = {**zero_vars, **initial_values}

    print("\nStarting minimization:")
    if not silent:
        print(f"circuit: {circuit}")
    start_time = time.time()
    result = tq.minimize(E, silent=True, initial_values=initial_values, *args, **kwargs)
    tot_time = time.time() - start_time

    # Print results
    if fci:
        error = result.energy - fci
        print(f"error: {error*1000}")
    print(f"time: {tot_time}s, {tot_time/60}m, {tot_time/60/60}h")
    if not silent:
        print(f"energy: {result.energy}")
        print("variables:", {**result.variables})
        print(f"num_variables: {len(circuit.extract_variables())}")
        if hasattr(circuit, "construct_circuit"):
            print(f"depth: {tq.compile_circuit(circuit.construct_circuit()).depth}")
        else:
            print(f"depth: {tq.compile_circuit(circuit).depth}")
        print(f"iterations: {result.history.iterations}")
        
    return result


def full_adapt_procedure(full_iterations, blockcircuit, operator_pool, H, fci, initial_values, ordering=["m0","a0","m1","a1"], silent=False, *args, **kwargs):
    # create an adapt procedure for different areas of the circuit in the order given by the list "ordering" and the position given by
    # codewords "m0" = after UC_mid of block0, "a0" = after UC after of block0, "m1" = after UC mid of block1 and so on
    circuit = copy.deepcopy(blockcircuit)
    
    print(f"Number of full iterations: {full_iterations}")
    print(f"Ordering: {ordering}")
    start_time = time.time()
    # each full_iter is a full calculation over all the elements in the ordering list
    for full_iter in range(full_iterations):
        for pos,j in enumerate(ordering):
            print(f"\nCurrent full_iter: {full_iter}, target: {j}")
            current_start_time = time.time()
            if full_iter==0 and pos==0:
                initial_values=initial_values
            else:
                initial_values=result.variables
            
            # Start adaptive method in spot "j"
            U, result = use_adapt(circuit, label=f"full_iter{full_iter},{j}", operator_pool=operator_pool, H=H,
                                  initial_values=initial_values, *args, **kwargs)
            error = result.energy - fci
            print(f"Current error: {error*1000}")
            current_tot_time = time.time() - current_start_time
            print(f"Current time: {current_tot_time}s, {current_tot_time/60}m, {current_tot_time/60/60}h")

            # Define new gates in qpic style
            adaptedU = []
            for gate in U.gates:
                if len(gate.indices)==1:
                    adaptedU.append(GenericGate(U=gate, name="single", n_qubits_is_double=True))
                if len(gate.indices)==2:
                    adaptedU.append(GenericGate(U=gate, name="double", n_qubits_is_double=True))
            
            # Add new gates to circuit
            if "m" in j:
                circuit[int(j[1])].add_UC_mid(adaptedU, overwrite=True)
            elif "a" in j:
                circuit[int(j[1])].add_UC_after(adaptedU, overwrite=True)

    tot_time = time.time() - start_time

    # Print results
    error = result.energy - fci
    print(f"Final error: {error*1000}")
    print(f"time: {tot_time}s, {tot_time/60}m, {tot_time/60/60}h")
    if not silent:
        print(f"energy: {result.energy}")
        print("variables:", {**result.variables})
        print(f"num_variables: {len(circuit.extract_variables())}")
        print(f"depth: {tq.compile_circuit(circuit.construct_circuit()).depth}")
    
    return circuit, result


def use_adapt(blockcircuit, label, operator_pool, H, initial_values, *args, **kwargs):
    # Define Upre and Upost for each spot
    if "m" in label:
        index = int(label[label.find("m")+1])
        Upre = blockcircuit.split(index)[0].construct_circuit()
        if blockcircuit[index].initial_state:
            Upre+= blockcircuit[index].get_UC_circuit()
        else:
            Upre+= blockcircuit[index].get_UR_circuit() + blockcircuit[index].get_UC_circuit()
        Upost = blockcircuit[index].get_UR_circuit().dagger() + blockcircuit[index].get_UC_after_circuit()
        Upost+= blockcircuit.split(index+1)[1].construct_circuit()

    elif "a" in label:
        index = int(label[label.find("a")+1])
        Upre, Upost = blockcircuit.split(index+1)
        Upre, Upost = Upre.construct_circuit(), Upost.construct_circuit()

    # Run adaptive method
    # print(f"Upre: {Upre}") # debug
    # print(f"Upost: {Upost}") # debug
    solver = tq.adapt.Adapt(H=H, Upre=Upre, Upost=Upost, operator_pool=operator_pool,
                            *args, **kwargs)
    result = solver(operator_pool=operator_pool, label=label, variables=initial_values)

    return result.U, result


def get_qpic(circuit, filename="circuit", png=False, pdf=False):
    print(f"Drawing {filename}_visual: {circuit}")
    if hasattr(circuit, "construct_visual"):
        export_to_qpic(circuit.construct_visual(), filename=f"{filename}_visual", filepath=os.getcwd())
    else:
        export_to_qpic(circuit, filename=f"{filename}_visual", filepath=os.getcwd())
    # circuit.construct_circuit().export_to(filename=f"{os.getcwd()}/{filename}.qpic")
    if png==True:
        qpic_to_png(filename=f"{filename}_visual")
    if pdf==True:
        qpic_to_pdf(filename=f"{filename}_visual")


def make_relax(label, molecule, name="butterfly"):
    relaxation = []

    if name=="test":
        relaxation.append(OrbitalRotatorGate(i=0, j=1, angle=(f"r01",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=2, j=3, angle=(f"r23",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=0, j=3, angle=(f"r03",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=1, j=2, angle=(f"r12",label), molecule=molecule))
        # relaxation.append(OrbitalRotatorGate(i=4, j=5, angle=(f"r45",label), molecule=molecule))
        # relaxation.append(OrbitalRotatorGate(i=1, j=2, angle=(f"r12",label), molecule=molecule))
        # relaxation.append(OrbitalRotatorGate(i=3, j=4, angle=(f"r34",label), molecule=molecule))

    elif name=="butterfly":
        relaxation.append(OrbitalRotatorGate(i=0, j=1, angle=(f"r01a",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=2, j=3, angle=(f"r23a",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=4, j=5, angle=(f"r45a",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=1, j=2, angle=(f"r12a",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=3, j=4, angle=(f"r34a",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=0, j=1, angle=(f"r01b",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=2, j=3, angle=(f"r23b",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=4, j=5, angle=(f"r45b",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=1, j=2, angle=(f"r12b",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=3, j=4, angle=(f"r34b",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=0, j=1, angle=(f"r01c",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=2, j=3, angle=(f"r23c",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=4, j=5, angle=(f"r45c",label), molecule=molecule))

    elif name=="symmetrized-1":
        relaxation.append(OrbitalRotatorGate(1,2,("ra",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(3,4,("rb",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(0,3,("rb",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(2,5,("ra",label), molecule=molecule))
        # relaxation.append(OrbitalRotatorGate(3,5,("rc",label), molecule=molecule))
        # relaxation.append(OrbitalRotatorGate(1,3,("rc",label, molecule=molecule))
        # relaxation.append(OrbitalRotatorGate(0,2,("rd",label), molecule=molecule))
        # relaxation.append(OrbitalRotatorGate(2,4,("rd",label), molecule=molecule))

    elif name=="symmetrized-2":
        relaxation.append(OrbitalRotatorGate(0,1,("ra",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(2,3,("rc",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(4,5,("rd",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(0,3,("ra",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(1,4,("rc",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(2,5,("rd",label), molecule=molecule))
        # relaxation.append(OrbitalRotatorGate(0,2,("rb",label), molecule=molecule))
        # relaxation.append(OrbitalRotatorGate(0,4,("rb",label), molecule=molecule))
        # relaxation.append(OrbitalRotatorGate(3,5,("re",label), molecule=molecule))
        # relaxation.append(OrbitalRotatorGate(1,5,("re",label), molecule=molecule))
        # relaxation.append(OrbitalRotatorGate(1,3,("rf",label), molecule=molecule))
        # relaxation.append(OrbitalRotatorGate(2,4,("rf",label), molecule=molecule))

    elif name=="1234":
        relaxation.append(OrbitalRotatorGate(i=0, j=1, angle=(f"r01a","g1",label), molecule=molecule)) # graph1a
        relaxation.append(OrbitalRotatorGate(i=2, j=3, angle=(f"r23a","g1",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=4, j=5, angle=(f"r45a","g1",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=1, j=2, angle=(f"r12a","g2",label), molecule=molecule)) # graph2a
        relaxation.append(OrbitalRotatorGate(i=3, j=4, angle=(f"r34a","g2",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=0, j=5, angle=(f"r05a","g2",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=0, j=3, angle=(f"r03a","g3",label), molecule=molecule)) # graph3a
        relaxation.append(OrbitalRotatorGate(i=1, j=2, angle=(f"r12a","g3",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=4, j=5, angle=(f"r45a","g3",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=2, j=5, angle=(f"r25a","g4",label), molecule=molecule)) # graph4a
        relaxation.append(OrbitalRotatorGate(i=0, j=1, angle=(f"r01a","g4",label), molecule=molecule))
        relaxation.append(OrbitalRotatorGate(i=3, j=4, angle=(f"r34a","g4",label), molecule=molecule))
    
    return relaxation


def make_correlators(label, molecule, name="butterfly"):
    UC = []

    if name=="butterfly":
        if "m0" not in label:
            UC.append(PairCorrelatorGate(i=0, j=1, angle=(f"c01a",label), molecule=molecule))
            UC.append(PairCorrelatorGate(i=2, j=3, angle=(f"c23a",label), molecule=molecule))
            UC.append(PairCorrelatorGate(i=4, j=5, angle=(f"c45a",label), molecule=molecule))
        UC.append(PairCorrelatorGate(i=1, j=2, angle=(f"c12a",label), molecule=molecule))
        UC.append(PairCorrelatorGate(i=3, j=4, angle=(f"c34a",label), molecule=molecule))
        UC.append(PairCorrelatorGate(i=0, j=1, angle=(f"c01b",label), molecule=molecule))
        UC.append(PairCorrelatorGate(i=2, j=3, angle=(f"c23b",label), molecule=molecule))
        UC.append(PairCorrelatorGate(i=4, j=5, angle=(f"c45b",label), molecule=molecule))
        UC.append(PairCorrelatorGate(i=1, j=2, angle=(f"c12b",label), molecule=molecule))
        UC.append(PairCorrelatorGate(i=3, j=4, angle=(f"c34b",label), molecule=molecule))
        UC.append(PairCorrelatorGate(i=0, j=1, angle=(f"c01c",label), molecule=molecule))
        UC.append(PairCorrelatorGate(i=2, j=3, angle=(f"c23c",label), molecule=molecule))
        UC.append(PairCorrelatorGate(i=4, j=5, angle=(f"c45c",label), molecule=molecule))

    elif name=="symmetrized-1":
        UC.append(PairCorrelatorGate(1,2,("ra",label), molecule=molecule))
        UC.append(PairCorrelatorGate(3,4,("rb",label), molecule=molecule))
        UC.append(PairCorrelatorGate(0,3,("rb",label), molecule=molecule))
        UC.append(PairCorrelatorGate(2,5,("ra",label), molecule=molecule))
        # UC.append(PairCorrelatorGate(3,5,("rc",label), molecule=molecule))
        # UC.append(PairCorrelatorGate(1,3,("rc",label), molecule=molecule))
        # UC.append(PairCorrelatorGate(0,2,("rd",label), molecule=molecule))
        # UC.append(PairCorrelatorGate(2,4,("rd",label), molecule=molecule))

    elif name=="symmetrized-2":
        UC.append(PairCorrelatorGate(0,1,("ra",label), molecule=molecule))
        UC.append(PairCorrelatorGate(2,3,("rc",label), molecule=molecule))
        UC.append(PairCorrelatorGate(4,5,("rd",label), molecule=molecule))
        UC.append(PairCorrelatorGate(0,3,("ra",label), molecule=molecule))
        UC.append(PairCorrelatorGate(1,4,("rc",label), molecule=molecule))
        UC.append(PairCorrelatorGate(2,5,("rd",label), molecule=molecule))
        # UC.append(PairCorrelatorGate(0,2,("rb",label), molecule=molecule))
        # UC.append(PairCorrelatorGate(0,4,("rb",label), molecule=molecule))
        # UC.append(PairCorrelatorGate(3,5,("re",label), molecule=molecule))
        # UC.append(PairCorrelatorGate(1,5,("re",label), molecule=molecule))
        # UC.append(PairCorrelatorGate(1,3,("rf",label), molecule=molecule))
        # UC.append(PairCorrelatorGate(2,4,("rf",label), molecule=molecule))

    elif name=="1234":
        if "m0" not in label:
            UC.append(PairCorrelatorGate(i=0, j=1, angle=(f"c01a","g1",label), molecule=molecule)) # graph1a
            UC.append(PairCorrelatorGate(i=2, j=3, angle=(f"c23a","g1",label), molecule=molecule))
            UC.append(PairCorrelatorGate(i=4, j=5, angle=(f"c45a","g1",label), molecule=molecule))
        UC.append(PairCorrelatorGate(i=1, j=2, angle=(f"c12a","g2",label), molecule=molecule)) # graph2a
        UC.append(PairCorrelatorGate(i=3, j=4, angle=(f"c34a","g2",label), molecule=molecule))
        UC.append(PairCorrelatorGate(i=0, j=5, angle=(f"c05a","g2",label), molecule=molecule))
        UC.append(PairCorrelatorGate(i=0, j=3, angle=(f"c03a","g3",label), molecule=molecule)) # graph3a
        UC.append(PairCorrelatorGate(i=1, j=2, angle=(f"c12a","g3",label), molecule=molecule))
        UC.append(PairCorrelatorGate(i=4, j=5, angle=(f"c45a","g3",label), molecule=molecule))
        UC.append(PairCorrelatorGate(i=2, j=5, angle=(f"c25a","g4",label), molecule=molecule)) # graph4a
        UC.append(PairCorrelatorGate(i=0, j=1, angle=(f"c01a","g4",label), molecule=molecule))
        UC.append(PairCorrelatorGate(i=3, j=4, angle=(f"c34a","g4",label), molecule=molecule))

    return UC


def fix_gnm(length, add_relax, relax_name):

    if length==1 and add_relax==False:
        print("One graph no relax")
        v = [-2.9143024926196883]
        tot_time = 1.728849172592163
        variables = {((0, 1), 'D', 0): -0.7159459176764383, ((2, 3), 'D', 0): -0.7062328473330444, ((4, 5), 'D', 0): -0.7159459176764315, ('main_r(4, 5)', 0): 0.004612584200824771, ('main_r(2, 3)', 0): -9.715010156237908e-12, ('main_r(0, 1)', 0): -0.004612584195447609, ('c', 0): 1.0}
    
    elif length==1 and add_relax==True and relax_name=="butterfly":
        print("One graph with relax-butterfly")
        v = [-2.9626811949915917]
        tot_time = 34.10075283050537
        variables = {((0, 1), 'D', 0): -0.6672458744250824, ((2, 3), 'D', 0): -0.6006607941051154, ((4, 5), 'D', 0): -0.6664776312685625, ('r45c', 0): 0.594935211030843, ('r23c', 0): -2.486094141727473, ('r01c', 0): 0.47794139522738505, ('r34b', 0): -0.4486828899849825, ('r12b', 0): -0.449599588569943, ('r45b', 0): -1.1550925610979224, ('r23b', 0): -1.1821165719183024, ('r01b', 0): -1.1335423041970383, ('r34a', 0): 0.42276567219525696, ('r12a', 0): 0.424119289931475, ('main_r(4, 5)', 0): 0.17302951625269514, ('main_r(2, 3)', 0): 1.153634633809399, ('main_r(0, 1)', 0): 0.2001573348838645, ('c', 0): 1.0}
    
    elif length==2 and add_relax==False:
        print("Two graphs no relax")
        v = [-2.9244191769866883]
        tot_time = 18.06506085395813
        variables = {((0, 1), 'D', 0): -0.7159459176764598, ((2, 3), 'D', 0): -0.7062328473330483, ((4, 5), 'D', 0): -0.715945917676414, ('main_r(4, 5)', 0): 0.004612584198916942, ('main_r(2, 3)', 0): -1.4602635777834823e-11, ('main_r(0, 1)', 0): -0.0046125841973535826, ((1, 2), 'D', 1): -0.7056822259209646, ((3, 4), 'D', 1): -0.7056822259223307, ((0, 5), 'D', 1): -1.5706484044886033, ('main_r(0, 5)', 1): -7.272034891589848e-09, ('main_r(3, 4)', 1): -0.0005340388933728319, ('main_r(1, 2)', 1): 0.0005340404774242743, ('c', 0): -0.9227913858329511, ('c', 1): -0.26800258108217795}
    
    elif length==2 and add_relax==True and relax_name=="butterfly":
        print("Two graphs with relax-butterfly")
        # v = [-2.976379181828332]
        # tot_time = 721.8532614707947
        # variables = {((0, 1), 'D', 0): -0.6766817546256761, ((2, 3), 'D', 0): -0.5877638492932803, ((4, 5), 'D', 0): -0.6632083286607781, ('r45c', 0): -4.8620793132987155, ('r23c', 0): 5.079367532916416, ('r01c', 0): -4.931447792922275, ('r34b', 0): -0.2395922246581582, ('r12b', 0): -0.2511867522203933, ('r45b', 0): -3.395388848555547, ('r23b', 0): 3.288753989790996, ('r01b', 0): -3.5758928413346216, ('r34a', 0): -0.19474566213014713, ('r12a', 0): -0.17053597299671056, ('main_r(4, 5)', 0): 2.631096133067214, ('main_r(2, 3)', 0): -2.660687804318174, ('main_r(0, 1)', 0): 2.7081226582023294, ((1, 2), 'D', 1): -0.3744293576200941, ((3, 4), 'D', 1): -0.3894117093006653, ((0, 5), 'D', 1): -1.6146669865685255, ('r45c', 1): -1.5413814933126873, ('r23c', 1): -0.28811403721403384, ('r01c', 1): -1.578924295971839, ('r34b', 1): -3.5821274989957845, ('r12b', 1): -4.333894049510276, ('r45b', 1): -2.1481632537297086, ('r23b', 1): 0.6479162571799724, ('r01b', 1): -2.3077067387403347, ('r34a', 1): 1.4496601795162616, ('r12a', 1): 1.113826410509273, ('r45a', 1): 1.80519716041598, ('r23a', 1): -0.22069643063756178, ('r01a', 1): 1.491101860498174, ('main_r(0, 5)', 1): -0.06294136917667716, ('main_r(3, 4)', 1): 0.9210907933205326, ('main_r(1, 2)', 1): 1.2147370092376242, ('c', 0): -0.8818855664365314, ('c', 1): -0.28817643019454975}
        v = [-2.976614569661678]
        tot_time =  392.44798159599304
        variables = {('r45c', 0): -3.283279934305668, ('r23c', 0): -0.6812788855088576, ('r01c', 0): -3.379049236541708, ('r34b', 0): 0.7670573196116482, ('r12b', 0): 0.7744679421945265, ('r45b', 0): 0.6424661693500765, ('r23b', 0): 0.7928107204658236, ('r01b', 0): 0.6298634647868305, ('r34a', 0): -0.6601079392209052, ('r12a', 0): -0.6714704246300617, ('r45c', 1): 2.654043095646404, ('r23c', 1): -1.9378524082374684, ('r01c', 1): 2.6630048596253486, ('r34b', 1): -0.49785702555492267, ('r12b', 1): -0.5000051288264745, ('r45b', 1): -1.00812071671092, ('r23b', 1): -0.06507471339955073, ('r01b', 1): -0.8958678226602104, ('r34a', 1): 0.2800403319691564, ('r12a', 1): 0.26401601478462067, ('r45a', 1): -2.126996210522077, ('r23a', 1): 1.8303250439538865, ('r01a', 1): -2.2396011000523828, ((0, 1), 'D', 0): -0.6517345400451329, ((2, 3), 'D', 0): -0.5602835073106275, ((4, 5), 'D', 0): -0.6537550486143169, ('main_r(4, 5)', 0): 0.85649106479218, ('main_r(2, 3)', 0): -2.238478948608373e-06, ('main_r(0, 1)', 0): 0.889353723418127, ((1, 2), 'D', 1): -0.4367559336788215, ((3, 4), 'D', 1): -0.43201115372888904, ((0, 5), 'D', 1): -1.8518494148775588, ('main_r(0, 5)', 1): -0.021610068598989115, ('main_r(3, 4)', 1): -0.029276039037521708, ('main_r(1, 2)', 1): -0.02344522718729187, ('c', 0): -0.9118234834592587, ('c', 1): -0.27230375797577694}

    elif length==4 and add_relax==False:
        print("Four graphs no relax")
        v = [-2.962671801993494]
        tot_time = 286.57593536376953
        variables = {((0, 1), 'D', 0): -0.672964997490187, ((2, 3), 'D', 0): -0.6019738382678731, ((4, 5), 'D', 0): -0.672965681240819, ('main_r(4, 5)', 0): 0.0020274725760761557, ('main_r(2, 3)', 0): 3.247356535659845e-10, ('main_r(0, 1)', 0): -0.002026605556980379, ((1, 2), 'D', 1): -0.1935807383969831, ((3, 4), 'D', 1): -0.19358509494488846, ((0, 5), 'D', 1): -1.5449268113647374, ('main_r(0, 5)', 1): -8.364593643335674e-06, ('main_r(3, 4)', 1): 0.004337636179405668, ('main_r(1, 2)', 1): -0.004333141632113156, ((0, 3), 'D', 2): -1.681565667218518, ((1, 2), 'D', 2): 0.011496294948008426, ((4, 5), 'D', 2): -0.5663625085294094, ('main_r(4, 5)', 2): 0.0021539635252343075, ('main_r(1, 2)', 2): 0.0002373256846190353, ('main_r(0, 3)', 2): 0.00023344244184032108, ((2, 5), 'D', 3): -1.6815642844074257, ((0, 1), 'D', 3): -0.5663628489047915, ((3, 4), 'D', 3): 0.011507130461481833, ('main_r(3, 4)', 3): -0.00023604351198369558, ('main_r(0, 1)', 3): -0.002150539156552706, ('main_r(2, 5)', 3): -0.00023736926831112936, ('c', 0): -0.7609714591679178, ('c', 1): -0.1417845643244215, ('c', 2): 0.23943513076852296, ('c', 3): 0.23943409637394653}
    
    elif length==4 and add_relax==True and relax_name=="butterfly":
        print("Four graphs with relax-butterfly")
        # v = [-2.9949226681690995]
        # tot_time = 7048.636281967163
        # variables = {((0, 1), 'D', 0): -0.636079079264088, ((2, 3), 'D', 0): -0.6410733219620027, ((4, 5), 'D', 0): -0.5907291087597092, ('r45c', 0): 0.33063865759958555, ('r23c', 0): 1.2142815571451075, ('r01c', 0): 0.22630775005375603, ('r34b', 0): -0.6605650396214935, ('r12b', 0): -0.04451475234517282, ('r45b', 0): 0.4752833595144103, ('r23b', 0): 0.23407833982753715, ('r01b', 0): 0.29221808709197467, ('r34a', 0): 0.37819771672703806, ('r12a', 0): -0.2864990393727234, ('main_r(4, 5)', 0): -0.2800623883933107, ('main_r(2, 3)', 0): -0.49441834871986196, ('main_r(0, 1)', 0): -0.18492764615075716, ((1, 2), 'D', 1): -0.3091029425128151, ((3, 4), 'D', 1): -0.31161594778178614, ((0, 5), 'D', 1): -1.6896726536956574, ('r45c', 1): 0.19733757377948694, ('r23c', 1): -0.25391397871189375, ('r01c', 1): -0.04414361961050247, ('r34b', 1): -2.3658430052504436, ('r12b', 1): -2.5462575995902808, ('r45b', 1): -0.8105002687773936, ('r23b', 1): -0.23312522129279123, ('r01b', 1): -0.7820569933106599, ('r34a', 1): -0.2704269948033203, ('r12a', 1): -0.9974822670201445, ('r45a', 1): 0.6616941897136838, ('r23a', 1): 0.480112200395034, ('r01a', 1): 0.6198032368982261, ('main_r(0, 5)', 1): -0.025031670421640683, ('main_r(3, 4)', 1): 0.8452906576545308, ('main_r(1, 2)', 1): 1.0789438476105153, ((0, 3), 'D', 2): -1.637073741792079, ((1, 2), 'D', 2): -0.17307157131604456, ((4, 5), 'D', 2): -0.5514386166810354, ('r45c', 2): -0.0009477314314643375, ('r23c', 2): 2.642844583068849, ('r01c', 2): 2.3440305918879174, ('r34b', 2): 0.23467649950740535, ('r12b', 2): -1.8508894063662322, ('r45b', 2): -1.3991128465070697, ('r23b', 2): -1.5936710950892732, ('r01b', 2): -1.2143471224872382, ('r34a', 2): -0.46291329567159134, ('r12a', 2): -0.1373088231273197, ('r45a', 2): 0.14810179369978377, ('r23a', 2): -1.252156280041754, ('r01a', 2): -1.2607268338525208, ('main_r(4, 5)', 2): 0.39111956816499016, ('main_r(1, 2)', 2): 0.19333724266120736, ('main_r(0, 3)', 2): -0.5918613253469779, ((2, 5), 'D', 3): -1.5134519595206568, ((0, 1), 'D', 3): -0.6057857129863398, ((3, 4), 'D', 3): -0.21688434866083744, ('r45c', 3): 0.2613459606461465, ('r23c', 3): 0.3628393520273899, ('r01c', 3): -1.7090408628589262, ('r34b', 3): -1.5321474965364947, ('r12b', 3): -0.0852869327728701, ('r45b', 3): -0.2415715870632702, ('r23b', 3): -1.2300177988534986, ('r01b', 3): -1.284023753963783, ('r34a', 3): 0.049470634821147476, ('r12a', 3): -0.191833348169675, ('r45a', 3): -0.21732099796285156, ('r23a', 3): 0.30855790119424625, ('r01a', 3): 0.35047642695752734, ('main_r(3, 4)', 3): 0.5313350494952601, ('main_r(0, 1)', 3): 0.8496348212523436, ('main_r(2, 5)', 3): -0.09598459884508724, ('c', 0): -0.7236296748423515, ('c', 1): -0.15700141405540874, ('c', 2): 0.2899622637183067, ('c', 3): 0.2466862137559418}
        v = [-2.995183448957649]
        tot_time = 13025.620218753815
        variables = {('r45c', 0): 0.9474036057071704, ('r23c', 0): 1.0227749198614737, ('r01c', 0): 0.5387756878575024, ('r34b', 0): -0.23671952352246098, ('r12b', 0): -1.0518788581316623, ('r45b', 0): -0.062101198290379196, ('r23b', 0): -0.32899926223191694, ('r01b', 0): 0.20895828807993577, ('r34a', 0): -0.08204900637149772, ('r12a', 0): 0.739085143818208, ('r45c', 1): -0.25335295724341433, ('r23c', 1): -0.17120527413216272, ('r01c', 1): 1.0234807862217057, ('r34b', 1): -2.114211466822059, ('r12b', 1): -1.998845358319817, ('r45b', 1): -0.8103735542975808, ('r23b', 1): -0.6453651927395404, ('r01b', 1): -1.6736333292423349, ('r34a', 1): 0.06620577159949363, ('r12a', 1): -1.5349292881696797, ('r45a', 1): 0.7913981801198763, ('r23a', 1): 0.6951961385648597, ('r01a', 1): 1.1494349224610412, ('r45c', 2): -1.4547862747523903, ('r23c', 2): 1.9985026489004805, ('r01c', 2): 2.291455132342685, ('r34b', 2): 0.18736499363892806, ('r12b', 2): -2.260283253427596, ('r45b', 2): -1.45018721703814, ('r23b', 2): -1.2150340507956172, ('r01b', 2): 0.7103284507680899, ('r34a', 2): -0.44443707362662743, ('r12a', 2): 0.7375303325759626, ('r45a', 2): 0.5747817542463558, ('r23a', 2): -1.4857553988673111, ('r01a', 2): -3.0003900225657256, ('r45c', 3): 2.5140065736924395, ('r23c', 3): 1.606115708099807, ('r01c', 3): -0.3585317320413311, ('r34b', 3): -1.9575901483370453, ('r12b', 3): -0.0693668708335746, ('r45b', 3): -1.1818601260541965, ('r23b', 3): -2.2412715289469696, ('r01b', 3): -0.3458495790102158, ('r34a', 3): 0.8554280085442129, ('r12a', 3): -0.25690593091627284, ('r45a', 3): -1.4582254681352, ('r23a', 3): 0.19655876255159221, ('r01a', 3): 0.22791697881508696, ((0, 1), 'D', 0): -0.5957773135132428, ((2, 3), 'D', 0): -0.6586200004491547, ((4, 5), 'D', 0): -0.6510856623075104, ('main_r(4, 5)', 0): -0.3065980332147053, ('main_r(2, 3)', 0): -0.2663702792413406, ('main_r(0, 1)', 0): -0.25816312894952237, ((1, 2), 'D', 1): -0.3219526099311122, ((3, 4), 'D', 1): -0.2846669010212937, ((0, 5), 'D', 1): -1.5350130352883364, ('main_r(0, 5)', 1): -0.007493979192277232, ('main_r(3, 4)', 1): 0.5930320555717434, ('main_r(1, 2)', 1): 1.0111910106656081, ((0, 3), 'D', 2): -1.6672200544888538, ((1, 2), 'D', 2): -0.1887837726103951, ((4, 5), 'D', 2): -0.6075126834204735, ('main_r(4, 5)', 2): 0.7566863816249777, ('main_r(1, 2)', 2): 0.2110602635400167, ('main_r(0, 3)', 2): -0.5115738051736579, ((2, 5), 'D', 3): -1.5218987597775686, ((0, 1), 'D', 3): -0.5725756345095134, ((3, 4), 'D', 3): -0.1863451688874785, ('main_r(3, 4)', 3): 0.034822641095556814, ('main_r(0, 1)', 3): 0.1389907672915441, ('main_r(2, 5)', 3): -0.5461561210853976, ('c', 0): -0.697511537450558, ('c', 1): -0.16001320166317123, ('c', 2): 0.2917838549606376, ('c', 3): 0.28100344836801133}

    return v, tot_time, variables
