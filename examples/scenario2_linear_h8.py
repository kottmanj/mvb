import tequila as tq
import numpy as np
import os, sys
import openfermion as of
import scipy

from mvb import Block, gates_to_orb_rot
from mvb import initialize_rotators, initialize_blockcircuit, optimize, get_qpic
from mvb import OrbitalRotatorGate, PairCorrelatorGate, GenericGate
from mvb import fold_rotators, get_one_body_operator, get_two_body_operator, get_hcb_part, rotate_and_hcb, compute_num_meas

import warnings
warnings.filterwarnings("ignore", category=tq.TequilaWarning)

geom = "H 0.0 0.0 0.0\nH 0.0 0.0 1.5\nH 0.0 0.0 3.0\nH 0.0 0.0 4.5\nH 0.0 0.0 6.0\nH 0.0 0.0 7.5\nH 0.0 0.0 9.0\nH 0.0 0.0 10.5"
mol = tq.Molecule(geometry=geom, basis_set="sto-3g").use_native_orbitals()
fci = mol.compute_energy("fci")
print(f"fci: {fci}")
H = mol.make_hamiltonian()

# Create true wave function and dummy circuit
# Hof = mol.make_hamiltonian().to_openfermion()
# Hsparse = of.linalg.get_sparse_operator(Hof)
# v,vv = scipy.sparse.linalg.eigsh(Hsparse, sigma=mol.compute_energy("fci"))
# print(f"error: {fci - v[0]}") # check
# print(f"vv[:,0]: {vv[:,0]}")
# x = vv[:,0]
# np.save('h8_wfn.npy', x)
x = np.load("h8_wfn.npy")
wfn = tq.QubitWaveFunction.from_array(x)
# print(f"wfn: {wfn}")

graphs = [
    [(0,1),(2,3),(4,5),(6,7)],
    [(0,7),(1,2),(3,4),(5,6)],
    [(0,1),(2,7),(3,6),(4,5)],
    [(0,3),(1,2),(4,7),(5,6)],
    [(0,5),(1,4),(2,3),(6,7)],
    [(0,7),(1,6),(2,5),(3,4)],
    # [(0,4),(1,3),(5,7),(2,6)],
    # [(0,2),(3,7),(4,6),(1,5)],
    # [(0,6),(1,5),(2,4),(3,7)]
]

# GNM preprocessing for Blockcircuit
relax = [[
    OrbitalRotatorGate(0,7,("r07",0)), OrbitalRotatorGate(1,2,("r12",0)), OrbitalRotatorGate(3,4,("r34",0)), OrbitalRotatorGate(5,6,("r56",0)),
    OrbitalRotatorGate(0,1,("r01",0)), OrbitalRotatorGate(2,3,("r23",0)), OrbitalRotatorGate(4,5,("r45",0)), OrbitalRotatorGate(6,7,("r67",0))
],[
    OrbitalRotatorGate(0,1,("r01",1)), OrbitalRotatorGate(2,3,("r23",1)), OrbitalRotatorGate(4,5,("r45",1)), OrbitalRotatorGate(6,7,("r67",1)),
    OrbitalRotatorGate(0,7,("r07",1)), OrbitalRotatorGate(1,2,("r12",1)), OrbitalRotatorGate(3,4,("r34",1)), OrbitalRotatorGate(5,6,("r56",1))
],[
    OrbitalRotatorGate(0,1,("r01",2)), OrbitalRotatorGate(2,3,("r23",2)), OrbitalRotatorGate(4,5,("r45",2)), OrbitalRotatorGate(6,7,("r67",2)),
    OrbitalRotatorGate(0,7,("r07",2)), OrbitalRotatorGate(1,2,("r12",2)), OrbitalRotatorGate(3,4,("r34",2)), OrbitalRotatorGate(5,6,("r56",2))
],[
    OrbitalRotatorGate(0,1,("r01",3)), OrbitalRotatorGate(2,3,("r23",3)), OrbitalRotatorGate(4,5,("r45",3)), OrbitalRotatorGate(6,7,("r67",3)),
    OrbitalRotatorGate(0,7,("r07",3)), OrbitalRotatorGate(1,2,("r12",3)), OrbitalRotatorGate(3,4,("r34",3)), OrbitalRotatorGate(5,6,("r56",3))
]]
_, variables, blocks = initialize_rotators(mol, graphs[:4], solve_GNM=False, add_relax=True, relax_name="custom", custom_relax=relax, silent=True)
variables = {((0, 1), 'D', 0): -0.7055232554659975, ((2, 3), 'D', 0): -0.730185065389639, ((4, 5), 'D', 0): -0.5585673944080015, ((6, 7), 'D', 0): -0.8715971053224815, ('r67', 0): 1.6433356258639098, ('r45', 0): -1.547743162812167, ('r23', 0): 1.5809681889687774, ('r01', 0): -1.5698893692667655, ('r56', 0): -0.20651621656306843, ('r34', 0): -0.29439905516255455, ('r12', 0): -0.20564909202125872, ('r07', 0): 0.022408372625257345, ('main_r(6, 7)', 0): -0.5238382278414893, ('main_r(4, 5)', 0): 0.4915094948252909, ('main_r(2, 3)', 0): -0.5021991418988903, ('main_r(0, 1)', 0): 0.4991928234122584, ((0, 7), 'D', 1): -1.2696834814568834, ((1, 2), 'D', 1): -0.5334982561829807, ((3, 4), 'D', 1): -0.3131639203362449, ((5, 6), 'D', 1): -0.6075957357717362, ('r56', 1): -1.5868682408533101, ('r34', 1): 4.194519749189254, ('r12', 1): -1.542677459828375, ('r07', 1): 1.5851583858338356, ('r67', 1): -2.7179474565360655, ('r45', 1): -0.05297685510048635, ('r23', 1): 0.062030744599300754, ('r01', 1): -2.6251692656186294, ('main_r(5, 6)', 1): 0.5154137413596065, ('main_r(3, 4)', 1): -1.333514094343005, ('main_r(1, 2)', 1): 0.4785616640412088, ('main_r(0, 7)', 1): -0.49865458131892726, ((0, 1), 'D', 2): -0.8801851485391283, ((2, 7), 'D', 2): -0.5437973515329, ((3, 6), 'D', 2): -3.17923172920524, ((4, 5), 'D', 2): -0.5244511547382409, ('r56', 2): -1.0140681422681765, ('r34', 2): -0.9799523859254043, ('r12', 2): -0.09443162687164423, ('r07', 2): -0.06752635750564738, ('r67', 2): 1.175945761194012, ('r45', 2): 0.010063571229552714, ('r23', 2): 1.1226587818476712, ('r01', 2): 0.004041644017033019, ('main_r(4, 5)', 2): -0.0013154401240812656, ('main_r(3, 6)', 2): 0.01077552034925381, ('main_r(2, 7)', 2): 0.004349612395276813, ('main_r(0, 1)', 2): 0.0016312638426463826, ((0, 3), 'D', 3): -2.1458403826085126, ((1, 2), 'D', 3): -0.0764894040101797, ((4, 7), 'D', 3): -2.287658418780267, ((5, 6), 'D', 3): -0.13863216045506066, ('r56', 3): -0.014964190303616743, ('r34', 3): 0.218829478879684, ('r12', 3): 0.011369017883979975, ('r07', 3): 0.21182856832102465, ('r67', 3): -0.968248465115223, ('r45', 3): -0.9703807056572098, ('r23', 3): -0.638379837930758, ('r01', 3): -0.6903302501439689, ('main_r(5, 6)', 3): 0.002538120414289311, ('main_r(4, 7)', 3): -0.0009015010384029253, ('main_r(1, 2)', 3): -0.0022506233072323765, ('main_r(0, 3)', 3): -0.0009327915481736972, ('c', 0): -0.6228389281483454, ('c', 1): 0.23010060119249226, ('c', 2): 0.16865713599187585, ('c', 3): -0.25785922492607405}
# energy = -3.968177509147281, error = 27.234192634891574

# result = optimize(blocks[0], H, fci, initial_values=variables, silent=False)
# variables1 = result.variables
# energy1 = result.energy
# Relaxed variables
energy1 = -3.9276003053457793
print("\nOne-graph")
print(f"error: {(energy1 - fci)*1000}")
variables1 = {((0, 1), 'D', 0): -0.6201344248602412, ((2, 3), 'D', 0): -0.6990220746286026, ((4, 5), 'D', 0): -0.5312990603202268, ((6, 7), 'D', 0): -0.714017875149138, ('r67', 0): 1.6425291982117702, ('r45', 0): -1.5528114813223377, ('r23', 0): 1.5728316853397872, ('r01', 0): -1.5670035271114664, ('r56', 0): -0.371814858568685, ('r34', 0): -0.37116012899303663, ('r12', 0): -0.36357690641768026, ('r07', 0): 0.0013316958515872755, ('main_r(6, 7)', 0): -0.5204363518211033, ('main_r(4, 5)', 0): 0.49424145434113487, ('main_r(2, 3)', 0): -0.5005621512850686, ('main_r(0, 1)', 0): 0.496186891664744, ((0, 7), 'D', 1): -1.2696834814568834, ((1, 2), 'D', 1): -0.5334982561829807, ((3, 4), 'D', 1): -0.3131639203362449, ((5, 6), 'D', 1): -0.6075957357717362, ('r56', 1): -1.5868682408533101, ('r34', 1): 4.194519749189254, ('r12', 1): -1.542677459828375, ('r07', 1): 1.5851583858338356, ('r67', 1): -2.7179474565360655, ('r45', 1): -0.05297685510048635, ('r23', 1): 0.062030744599300754, ('r01', 1): -2.6251692656186294, ('main_r(5, 6)', 1): 0.5154137413596065, ('main_r(3, 4)', 1): -1.333514094343005, ('main_r(1, 2)', 1): 0.4785616640412088, ('main_r(0, 7)', 1): -0.49865458131892726, ((0, 1), 'D', 2): -0.8801851485391283, ((2, 7), 'D', 2): -0.5437973515329, ((3, 6), 'D', 2): -3.17923172920524, ((4, 5), 'D', 2): -0.5244511547382409, ('r56', 2): -1.0140681422681765, ('r34', 2): -0.9799523859254043, ('r12', 2): -0.09443162687164423, ('r07', 2): -0.06752635750564738, ('r67', 2): 1.175945761194012, ('r45', 2): 0.010063571229552714, ('r23', 2): 1.1226587818476712, ('r01', 2): 0.004041644017033019, ('main_r(4, 5)', 2): -0.0013154401240812656, ('main_r(3, 6)', 2): 0.01077552034925381, ('main_r(2, 7)', 2): 0.004349612395276813, ('main_r(0, 1)', 2): 0.0016312638426463826, ((0, 3), 'D', 3): -2.1458403826085126, ((1, 2), 'D', 3): -0.0764894040101797, ((4, 7), 'D', 3): -2.287658418780267, ((5, 6), 'D', 3): -0.13863216045506066, ('r56', 3): -0.014964190303616743, ('r34', 3): 0.218829478879684, ('r12', 3): 0.011369017883979975, ('r07', 3): 0.21182856832102465, ('r67', 3): -0.968248465115223, ('r45', 3): -0.9703807056572098, ('r23', 3): -0.638379837930758, ('r01', 3): -0.6903302501439689, ('main_r(5, 6)', 3): 0.002538120414289311, ('main_r(4, 7)', 3): -0.0009015010384029253, ('main_r(1, 2)', 3): -0.0022506233072323765, ('main_r(0, 3)', 3): -0.0009327915481736972, ('c', 0): -0.6228389281483454, ('c', 1): 0.23010060119249226, ('c', 2): 0.16865713599187585, ('c', 3): -0.25785922492607405}

# blockcircuit = initialize_blockcircuit(mol, graphs[:2], input_blocks=blocks)
# result = optimize(blockcircuit, H, fci, initial_values=result.variables, silent=False)
# variables2 = result.variables
# energy2 = result.energy
# Relaxed variables
energy2 = -3.960390565925205
print("\nTwo-graph")
print(f"error: {(energy2 - fci)*1000}")
variables2 = {((0, 1), 'D', 0): -3.679683102792911, ((2, 3), 'D', 0): -0.7249984703179383, ((4, 5), 'D', 0): -0.17408857793834026, ((6, 7), 'D', 0): -0.7315176628841076, ('r67', 0): 1.444270635698056, ('r45', 0): -1.7914720228690701, ('r23', 0): 1.7164410746428012, ('r01', 0): -1.9804552167297198, ('r56', 0): -0.4576366904881185, ('r34', 0): -0.41513182215221334, ('r12', 0): -0.09184888517209326, ('r07', 0): -0.017949882221657037, ('main_r(6, 7)', 0): -0.4596475776544337, ('main_r(4, 5)', 0): 0.5882175331537645, ('main_r(2, 3)', 0): -0.5443893708615801, ('main_r(0, 1)', 0): 0.7852713564193569, ('main_r(0, 7)', 1): -0.49977353245645867, ('main_r(1, 2)', 1): 0.5006197244558145, ('main_r(3, 4)', 1): -0.4653041375160603, ('main_r(5, 6)', 1): 0.558730460517907, ('r01', 1): -2.788154523511555, ('r23', 1): 0.019039003395197724, ('r45', 1): 2.3391423649489336, ('r67', 1): -0.11322292798484165, ('r07', 1): 1.6086676997253049, ('r12', 1): -1.0977278902566339, ('r34', 1): 1.1269144521224017, ('r56', 1): -2.0174531024175892, ((0, 7), 'c', 1): 0.033404066445946554, ((1, 2), 'c', 1): 3.4404246651007675, ((3, 4), 'c', 1): -0.060071843253182364, ((5, 6), 'c', 1): 0.7500783882530436, ((0, 7), 'D', 1): -1.2696834814568834, ((1, 2), 'D', 1): -0.5334982561829807, ((3, 4), 'D', 1): -0.3131639203362449, ((5, 6), 'D', 1): -0.6075957357717362, ((0, 1), 'D', 2): -0.8801851485391283, ((2, 7), 'D', 2): -0.5437973515329, ((3, 6), 'D', 2): -3.17923172920524, ((4, 5), 'D', 2): -0.5244511547382409, ('r56', 2): -1.0140681422681765, ('r34', 2): -0.9799523859254043, ('r12', 2): -0.09443162687164423, ('r07', 2): -0.06752635750564738, ('r67', 2): 1.175945761194012, ('r45', 2): 0.010063571229552714, ('r23', 2): 1.1226587818476712, ('r01', 2): 0.004041644017033019, ('main_r(4, 5)', 2): -0.0013154401240812656, ('main_r(3, 6)', 2): 0.01077552034925381, ('main_r(2, 7)', 2): 0.004349612395276813, ('main_r(0, 1)', 2): 0.0016312638426463826, ((0, 3), 'D', 3): -2.1458403826085126, ((1, 2), 'D', 3): -0.0764894040101797, ((4, 7), 'D', 3): -2.287658418780267, ((5, 6), 'D', 3): -0.13863216045506066, ('r56', 3): -0.014964190303616743, ('r34', 3): 0.218829478879684, ('r12', 3): 0.011369017883979975, ('r07', 3): 0.21182856832102465, ('r67', 3): -0.968248465115223, ('r45', 3): -0.9703807056572098, ('r23', 3): -0.638379837930758, ('r01', 3): -0.6903302501439689, ('main_r(5, 6)', 3): 0.002538120414289311, ('main_r(4, 7)', 3): -0.0009015010384029253, ('main_r(1, 2)', 3): -0.0022506233072323765, ('main_r(0, 3)', 3): -0.0009327915481736972, ('c', 0): -0.6228389281483454, ('c', 1): 0.23010060119249226, ('c', 2): 0.16865713599187585, ('c', 3): -0.25785922492607405}

blockcircuit = initialize_blockcircuit(mol, graphs[:4], input_blocks=blocks)
# result = optimize(blockcircuit, H, fci, initial_values=result.variables, silent=False)
# variables2 = result.variables
# energy2 = result.energy
# Relaxed variables
energy2 = -3.9802884031927466
print("\nFour-graph")
print(f"error: {(energy2 - fci)*1000}")
variables2 = {((0, 1), 'D', 0): -4.122086280629323, ((2, 3), 'D', 0): -0.8445785623883298, ((4, 5), 'D', 0): 0.4117397552800672, ((6, 7), 'D', 0): -0.6998348754746423, ('r67', 0): 1.5923779812136487, ('r45', 0): -1.6145402890821772, ('r23', 0): 1.5092398008941799, ('r01', 0): -1.2549338464716382, ('r56', 0): -0.5765731746026791, ('r34', 0): -0.4652195840479327, ('r12', 0): 0.5283347411174105, ('r07', 0): -0.03082151099784686, ('main_r(6, 7)', 0): -0.5025540189550615, ('main_r(4, 5)', 0): 0.5557288974292617, ('main_r(2, 3)', 0): -0.4745531132290902, ('main_r(0, 1)', 0): 0.5986074379214785, ('main_r(0, 7)', 1): -0.5002097081139728, ('main_r(1, 2)', 1): 0.4484169997350065, ('main_r(3, 4)', 1): -0.4664221569847796, ('main_r(5, 6)', 1): 0.6073058215008499, ('r01', 1): -2.417573147551695, ('r23', 1): -0.18869279681786735, ('r45', 1): 2.666918784776019, ('r67', 1): -0.24535927081675318, ('r07', 1): 1.6993585771260262, ('r12', 1): -1.3622367078186226, ('r34', 1): 1.684331005671337, ('r56', 1): -1.9235305076181057, ((0, 7), 'c', 1): -0.01913489136230011, ((1, 2), 'c', 1): 2.8212690431108864, ((3, 4), 'c', 1): -0.258744862085965, ((5, 6), 'c', 1): 1.5507460079225361, ('main_r(0, 1)', 2): 0.1649409571580741, ('main_r(2, 7)', 2): -0.3494823311545557, ('main_r(3, 6)', 2): 0.3689157449481716, ('main_r(4, 5)', 2): 0.6362982882293433, ('r01', 2): 0.056024733909178295, ('r23', 2): 0.4908818381173156, ('r45', 2): 0.21302232453197012, ('r67', 2): 1.378530005695309, ('r07', 2): -0.05582130095956819, ('r12', 2): -0.09738598931853042, ('r34', 2): -1.1741613212783353, ('r56', 2): -0.5006566780367068, ((0, 1), 'c', 2): 1.4791907957785833, ((2, 7), 'c', 2): -0.009607654275636287, ((3, 6), 'c', 2): 0.030583956317571298, ((4, 5), 'c', 2): -0.4397177474043326, ('main_r(0, 3)', 3): -0.016836828069104796, ('main_r(1, 2)', 3): 0.04349064350426613, ('main_r(4, 7)', 3): 0.12198314008850109, ('main_r(5, 6)', 3): 0.01162451742091935, ('r01', 3): 0.18856798897414312, ('r23', 3): 0.39637650571804817, ('r45', 3): 0.13060619019494824, ('r67', 3): -0.21001095767548766, ('r07', 3): -0.25307698341609786, ('r12', 3): -0.284534159287695, ('r34', 3): -0.4141497227688098, ('r56', 3): -0.2382360553133213, ((0, 3), 'c', 3): -0.04886191706732704, ((1, 2), 'c', 3): -1.0922628833652412, ((4, 7), 'c', 3): 0.2592494663397049, ((5, 6), 'c', 3): -0.4076084869946667, ((0, 7), 'D', 1): -1.2696834814568834, ((1, 2), 'D', 1): -0.5334982561829807, ((3, 4), 'D', 1): -0.3131639203362449, ((5, 6), 'D', 1): -0.6075957357717362, ((0, 1), 'D', 2): -0.8801851485391283, ((2, 7), 'D', 2): -0.5437973515329, ((3, 6), 'D', 2): -3.17923172920524, ((4, 5), 'D', 2): -0.5244511547382409, ((0, 3), 'D', 3): -2.1458403826085126, ((1, 2), 'D', 3): -0.0764894040101797, ((4, 7), 'D', 3): -2.287658418780267, ((5, 6), 'D', 3): -0.13863216045506066, ('c', 0): -0.6228389281483454, ('c', 1): 0.23010060119249226, ('c', 2): 0.16865713599187585, ('c', 3): -0.25785922492607405}

# Create rotators
# URs, variables, blocks, v = initialize_rotators(mol, graphs[:4], solve_GNM=True, add_relax=False, silent=True)
URs, variables, blocks = initialize_rotators(mol, graphs[:6], solve_GNM=False, add_relax=False, silent=True)
variables = {}
for block in blocks:
    variables.update({v:0 for v in block.extract_variables() if "r" in str(v)})

debug = False
if debug:
    print("------------------------------")
    print("TEST 1")
    # transform both H and U
    UR = blocks[0].get_UR_circuit().map_variables(variables)
    HX = fold_rotators(mol, UR).make_hamiltonian()
    EX = tq.ExpectationValue(H=HX, U=blockcircuit.construct_circuit()+UR)
    print(f"transform both H and U: {tq.simulate(EX, variables2)}")
    print(f"result.energy:          {energy2}")
    print(energy2 - tq.simulate(EX, variables2))
    print(np.isclose(energy2, tq.simulate(EX, variables2)))
    print("done")

    print("TEST 2")
    # sum of E1 and E2 is the same as E
    H1 = get_one_body_operator(mol)
    H2 = get_two_body_operator(mol)
    E1 = tq.ExpectationValue(U=blockcircuit.construct_circuit(), H=H1)
    print(f"E1: {tq.simulate(E1, variables=variables2)}")
    E2 = tq.ExpectationValue(U=blockcircuit.construct_circuit(), H=H2)
    print(energy2 - tq.simulate(E1 + E2, variables=variables2))
    print(f"result.energy == E1+E2: {np.isclose(energy2,tq.simulate(E1 + E2, variables=variables2))}")
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
    Block(UR=blocks[2].UR).map_variables(variables),
    Block(UR=blocks[3].UR).map_variables(variables),
    Block(UR=blocks[4].UR).map_variables(variables),
    Block(UR=blocks[5].UR).map_variables(variables)
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

    # measure H rotated in SPA with SPA circuit
    print("TEST 4")
    E_test = tq.ExpectationValue(U=blocks[0].construct_circuit(), H=mol.make_hamiltonian())
    energy1_test = tq.simulate(E_test, variables=variables1)
    print(f"energy1_test: {energy1_test}")
    tmol = fold_rotators(mol, blocks[0].get_UR_circuit().map_variables(variables1))
    tcircuit = blocks[0].construct_circuit() + blocks[0].get_UR_circuit().map_variables(variables1)
    hcb_mol, non_hcb_mol = get_hcb_part(tmol)
    EX = tq.ExpectationValue(U=tcircuit, H=hcb_mol.make_hamiltonian())
    incr = tq.simulate(EX, variables=variables1)
    approx += incr
    print(f"incr:               {incr}")
    print(f"error in new basis: {energy1_test-approx}")
    print(f"new approx:         {approx}")
    print("### 'error in new basis' should be zero ###")
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
approx += incr
circuit_wfn = tq.simulate(blockcircuit.construct_circuit(), variables=variables2)
M_tot = compute_num_meas(circuit_wfn, is_hcb=True, hcb_mol=hcb_mol, debug=False, n_repetitions=100)
print(f"M_tot:              {M_tot:e}")
# if debug:
#     E_shots = tq.simulate(EX, variables=variables2, samples=int(M_tot), backend="qulacs")
#     print(f"Energy with shots: {E_shots}")
#     print(f"Error with shots: {(E_shots-incr):e}")
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
                                       old_UR=old_UR, debug=debug, true_wfn=wfn, n_repetitions=100)
    old_UR = rotation.get_UR_circuit()
    errors.append(target-approx)
    print("------------------------------")

print(f"errors = {errors}")
