from typing import Any
import tequila as tq
from tequila import QCircuit
import numpy as np
import copy


class Block:
    UR = None
    UC = None
    UC_after = None
    initial_state: bool = False
    dagger: bool = True

    def __init__(self, UR=None, UC=None, UC_after=None, initial_state=False, dagger=True):
        if UR:
            if isinstance(UR, list):
                self.UR = UR
            elif hasattr(UR, "gates"):
                self.UR = [g for g in UR.gates]
            else:
                self.UR = [UR]
        else:
            self.UR = []

        if UC:
            if isinstance(UC, list):
                self.UC = UC
            elif hasattr(UC, "gates"):
                self.UC = [g for g in UC.gates]
            else:
                self.UC = [UC]
        else:
            self.UC = []

        if UC_after:
            if isinstance(UC_after, list):
                self.UC_after = UC_after
            elif hasattr(UC_after, "gates"):
                self.UC_after = [g for g in UC_after.gates]
            else:
                self.UC_after = [UC_after]
        else:
            self.UC_after = []
        self.initial_state = initial_state
        self.dagger = dagger

    # Construct a list of gates in the structured block order
    # Meant for gates from qpic_visualization.py
    def construct_visual(self):
        if self.initial_state:
            return self.UC[:] + self.UR[::-1] + self.UC_after[:]
        if not self.dagger:
            return self.UR[:] + self.UC[:] + self.UC_after[:]
        else:
            return self.UR[:] + self.UC[:] + self.UR[::-1] + self.UC_after[:]
    
    # Construct UR in the tq.QCircuit() form
    def get_UR_circuit(self):
        UR = tq.QCircuit()
        for rot in self.UR:
            if hasattr(rot, "construct_circuit"):
                UR += rot.construct_circuit()
            else:
                UR += rot
        return UR
    
    # Construct UC in the tq.QCircuit() form
    def get_UC_circuit(self):
        UC = tq.QCircuit()
        for corr in self.UC:
            if hasattr(corr, "construct_circuit"):
                UC += corr.construct_circuit()
            else:
                UC += corr
        return UC
    
    # Construct UC_after in the tq.QCircuit() form
    def get_UC_after_circuit(self):
        UC_after = tq.QCircuit()
        for corr_after in self.UC_after:
            if hasattr(corr_after, "construct_circuit"):
                UC_after += corr_after.construct_circuit()
            else:
                UC_after += corr_after
        return UC_after
    
    # Construct the full Block in the tq.QCircuit() form
    def construct_circuit(self):
        UR = self.get_UR_circuit()
        UC = self.get_UC_circuit()
        UC_after = self.get_UC_after_circuit()

        if self.initial_state:
            return UC + UR.dagger() + UC_after
        if not self.dagger:
            return UR + UC + UC_after
        else:
            return UR + UC + UR.dagger() + UC_after

    def __call__(self, *args, **kwargs):
        return self.construct_circuit()

    def __str__(self):
        return f"I am a Block with variables: {self.construct_circuit().extract_variables()}"
    
    # Add gates to the UR list
    def add_UR(self, UR, overwrite=False):
        if overwrite:
            if isinstance(UR, list):
                self.UR.extend(UR)
            else:
                self.UR.extend([g for g in UR.gates])
        else:
            block = copy.deepcopy(self)
            if isinstance(UR, list):
                block.UR.extend(UR)
            else:
                block.UR.extend([g for g in UR.gates])
            return block

    # Add gates to the UC list
    def add_UC_mid(self, UC, overwrite=False):
        if overwrite:
            if isinstance(UC, list):
                self.UC.extend(UC)
            else:
                self.UC.extend([g for g in UC.gates])
        else:
            block = copy.deepcopy(self)
            if isinstance(UC, list):
                block.UC.extend(UC)
            else:
                block.UC.extend([g for g in UC.gates])
            return block


    # Add gates to the UC_after list
    def add_UC_after(self, UC_after, overwrite=False):
        if overwrite:
            if isinstance(UC_after, list):
                self.UC_after.extend(UC_after)
            else:
                self.UC_after.extend([g for g in UC_after.gates])
        else:
            block = copy.deepcopy(self)
            if isinstance(UC_after, list):
                block.UC_after.extend(UC_after)
            else:
                block.UC_after.extend([g for g in UC_after.gates])
            return block

    def extract_variables(self):
        return self.construct_circuit().extract_variables()
    
    def map_variables(self, variables: dict, *args, **kwargs):
        mapped_block = copy.deepcopy(self)

        new_UR = []
        for gate in mapped_block.UR:
            new_UR.append(gate.map_variables(variables))

        new_UC = []
        for gate in mapped_block.UC:
            new_UC.append(gate.map_variables(variables))

        new_UC_after = []
        for gate in mapped_block.UC_after:
            new_UC_after.append(gate.map_variables(variables))

        new_block = Block(UR=new_UR, UC=new_UC, UC_after=new_UC_after, initial_state=mapped_block.initial_state, dagger=mapped_block.dagger)

        return new_block


class BlockCircuit:
    blocks: list = None

    def __init__(self, blocks=None):
        if blocks:
            if isinstance(blocks, list):
                self.blocks = blocks
            else:
                self.blocks = [blocks]
        else:
            self.blocks = []

    @property
    def gates(self):
        return self.construct_circuit().gates

    def __add__(self,other:Block):
        result = BlockCircuit(copy.deepcopy(self.blocks))
        if isinstance(other, Block):
            result.blocks.append(other)
        else:
            result.blocks += other.blocks
        return result
    
    def __getitem__(self, i):
        return self.blocks[i]
    
    # def __setitem__(self, key, value):
    #     assert hasattr(value, "construct_circuit")
    #     assert callable(value)
    #     self.blocks[key] = value
    
    def split(self, i):
        '''
        i refers to the first index of the right block
        or the last one (excluded) of the left one.
        Same as in list slicing.
        '''
        left = BlockCircuit(self.blocks[:i])
        right = BlockCircuit(self.blocks[i:])
        return left, right

    # Construct a list of gates in the structured block order
    def construct_visual(self):
        return sum([block.construct_visual() for block in self.blocks], [])

    # Construct the full BlockCircuit in the tq.QCircuit() form
    def construct_circuit(self):
        return sum([block.construct_circuit() for block in self.blocks], tq.QCircuit())
    
    def __call__(self, *args, **kwargs):
        return self.construct_circuit()

    def __str__(self):
        return f"I am a BlockCircuit with variables: {self.construct_circuit().extract_variables()}"
    
    def extract_variables(self):
        return self.construct_circuit().extract_variables()
    
    def map_variables(self, variables: dict, *args, **kwargs):
        mapped_blockcircuit = copy.deepcopy(self)
        new_blocks = []
        for block in mapped_blockcircuit.blocks:
            new_blocks.append(block.map_variables(variables))

        new_blockcircuit = BlockCircuit(blocks=new_blocks)

        return new_blockcircuit


class BlockPool(tq.adapt.AdaptPoolBase):

    def make_unitary(self, k, label) -> QCircuit:
        U = self.generators[k]
        keymap = {var:(var, k, label) for var in U.extract_variables()}
        U = U.map_variables(keymap)
        return U
    
    
class ObjectiveFactoryBlock(tq.adapt.ObjectiveFactoryBase):
    
    def __init__(self, H=None, Upre=None, Upost=None, *args, **kwargs):
        super().__init__(H, Upre, Upost, *args, **kwargs)
        if hasattr(self.Upre, "construct_circuit"):
            self.Upre = self.Upre.construct_circuit()
        if hasattr(self.Upost, "construct_circuit"):
            self.Upost = self.Upost.construct_circuit()

    def __call__(self, U, *args, **kwargs):
        
        if hasattr(U, "construct_circuit"):
            UX = U.construct_circuit()
        else:
            UX = U

        return tq.ExpectationValue(H=self.H, U=self.Upre + UX + self.Upost)
    
def gates_to_orb_rot(rotation_circuit, dim):

    # Extract parameters and indices from gates
    params = []
    indices = []
    for gate in reversed(rotation_circuit.gates):
        params.append(gate.parameter)
        indices.append(gate.indices)

    # Select only even values
    # Because UR gates rotate molecular orbitals 
    params = params[::2]
    indices = indices[::2]

    # Refactor as a list
    tmp = []
    for i in indices:
        tmp.append((int(i[0][0]/2),int(i[0][1]/2)))
    indices = tmp

    # Create a matrix for each UR gate
    rot_matrices = []
    for k,index in enumerate(indices):
        tmp = np.eye(dim) # dim is the dimension of orbital coefficients
        if isinstance(params[k], tq.Objective):
            params[k] = params[k]()
        tmp[index[0]][index[0]] = np.cos(params[k]/2)
        tmp[index[0]][index[1]] = np.sin(params[k]/2)
        tmp[index[1]][index[0]] = -np.sin(params[k]/2)
        tmp[index[1]][index[1]] = np.cos(params[k]/2)
        rot_matrices.append(tmp)

    # Multiply all matrices
    tmp = rot_matrices[0]
    for matrix in rot_matrices[1:]:
        tmp = np.dot(tmp,matrix)
    transformation_matrix = tmp

    return transformation_matrix