import abc
from typing import List

import numpy
import numbers
import tequila as tq
import subprocess
import os
from math import pi, floor
import copy


class Gate(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def construct_circuit(self) -> tq.QCircuit:
        pass

    @abc.abstractmethod
    def map_variables(self, variables) -> "Gate":
        pass


class RenderableGate(Gate, metaclass=abc.ABCMeta):
    """
    Gate that can generate it's qpic visulization on its own
    """

    @abc.abstractmethod
    def render_circuit(self) -> str:
        """
        returns qpic string, tailing \n is not needed!
        """
        pass

    @abc.abstractmethod
    def used_wires(self) -> List[int]:
        """
        all qubits that are used by this gate
        """
        pass


class PairCorrelatorGate(Gate):
    '''
        i,j correspond to Molecular Orbital index --> ((2*i,2*j),(2*i+1,2*j+1))
    '''

    def __init__(self, i, j, angle, control=None, assume_real=True, encoding="JordanWigner", unit_of_pi: bool = False,
                 *args, **kwargs):
        self.i = i
        self.j = j
        self.angle = tq.assign_variable(angle)
        self.unit_of_pi = unit_of_pi
        self.control = control
        self.assume_real = assume_real
        if "molecule" not in kwargs:
            k = max(self.i, self.j)
            x = numpy.zeros(k ** 2).reshape([k, k])
            y = numpy.zeros(k ** 4).reshape([k, k, k, k])
            if encoding is None:
                encoding = "jordanwigner"
            self.encoding = encoding.lower()
            if "molecule_factory" not in kwargs:
                self._dummy = tq.Molecule(geometry="", one_body_integrals=x, two_body_integrals=y,
                                          nuclear_repulsion=0.0,
                                          transformation=self.encoding, *args, **kwargs)
            else:
                mol_fact = kwargs["molecule_factory"]
                kwargs.pop("molecule_factory")
                self._dummy = mol_fact(geometry="", one_body_integrals=x, two_body_integrals=y, nuclear_repulsion=0.0,
                                       transformation=self.encoding, *args, **kwargs)
        else:
            self._dummy = kwargs["molecule"]
            self.encoding = self._dummy.transformation
            kwargs.pop("molecule")

    def construct_circuit(self, *args, **kwargs):
        if self.encoding == "jordanwigner":
            return tq.gates.QubitExcitation(angle=self.angle, control=self.control,
                                            target=[2 * self.i, 2 * self.j, 2 * self.i + 1, 2 * self.j + 1],
                                            assume_real=self.assume_real)
        else:
            return self._dummy.UC(self.i, self.j, control=self.control, assume_real=self.assume_real, angle=self.angle,
                                  *args, **kwargs)

    def map_variables(self, variables):
        mapped_gate = copy.deepcopy(self)
        mapped_gate.angle = mapped_gate.angle.map_variables(variables)
        return mapped_gate


class OrbitalRotatorGate(PairCorrelatorGate):
    '''
        i,j correspond to Molecular Orbital index --> ((2*i,2*j),(2*i+1,2*j+1))
    '''

    def construct_circuit(self):
        return self._dummy.UR(self.i, self.j, control=self.control, assume_real=self.assume_real, angle=self.angle)


class Single_excitation(PairCorrelatorGate):
    '''
        i,j correspond to Spin Orbital index --> ((i,j))
    '''

    def construct_circuit(self, *args, **kwargs):
        return self._dummy.make_excitation_gate(indices=((self.i, self.j)), angle=self.angle,
                                                assume_real=self.assume_real, control=self.control, *args, **kwargs)


class Double_excitation(PairCorrelatorGate):
    '''
            i,j,k,l correspond to Spin Orbital index --> ((i,j),(k,l))
    '''

    def __init__(self, i, j, k, l, angle, control=None, assume_real=True, encoding="JordanWigner", *args, **kwargs):
        super().__init__(i=i, j=j, angle=angle, control=control, assume_real=assume_real, encoding=encoding, *args,
                         **kwargs)
        self.k = k
        self.l = l
        if self.i // 2 == self.k // 2 and self.j // 2 == self.l // 2:
            self.type = 0  # Actually its a UC
        elif self.i // 2 == self.k // 2 and self.j // 2 != self.l // 2:
            self.type = 1  # Origin Orbital Paired, Destination Unpaired
        elif self.i // 2 != self.k // 2 and self.j // 2 == self.l // 2:
            self.type = 2  # Origin Orbital Unpaired, Destiation Paired
        else:
            self.type = 3  # Completily Unpaired

    def construct_circuit(self, *args, **kwargs):
        if not self.type:
            return PairCorrelatorGate(i=self.i // 2, j=self.j // 2, assume_real=self.assume_real, angle=self.angle,
                                      encoding=self.encoding).construct_circuit(*args, **kwargs)
        else:
            return self._dummy.make_excitation_gate(indices=((self.i, self.j), (self.k, self.l)),
                                                    assume_real=self.assume_real, angle=self.angle,
                                                    control=self.control, *args, **kwargs)


class GenericGate:
    'Trotter gates are left just as U.export_to(name.qpic), take care of the qubits yourself'

    # TODO: maybe something can be done with the Trotter gates?
    def __init__(self, U, name=None, n_qubits_is_double=False, *args, **kwargs):
        self.U = U
        name_opt = ["initialstate", "simple", "single", "double", "trotter"]
        if name.lower() in name_opt:
            self.name = name.lower()
        else:
            self.name = "simple"
        self.qubits = []
        if name == "single" or name == "double":
            for index in U.indices:
                for i in index:
                    self.qubits.append(i)
        elif name == 'trotter':
            self.qubits.extend(U._target)
            self.qubits = list(set(self.qubits))
        else:
            for gate in self.U.gates:
                self.qubits.extend(gate.qubits)
            self.qubits = list(set(self.qubits))  # remove duplicates
        if n_qubits_is_double:
            spatial = [q // 2 for q in self.qubits]  # half the number of qubits visualized
            self.qubits = list(set(spatial))

    def construct_circuit(self):
        return self.U

    def map_variables(self, variables):
        mapped_gate = copy.deepcopy(self)
        for g in mapped_gate.construct_circuit().gates:
            g = g.map_variables(variables)
        return mapped_gate


def export_to_qpic(list_of_gates, filename=None, filepath=None,
                   group_together=False, qubit_names=None, mark_parametrized_gates=False, color_range: bool = False,
                   gatecolor1="tq",
                   textcolor1="white", gatecolor2="fai", textcolor2="white", gatecolor3="unia", textcolor3="black",
                   color_from='blue', color_to='red', *args, **kwargs) -> str:
    result = ""

    colors = [{"name": "tq", "rgb": (0.03137254901960784, 0.1607843137254902, 0.23921568627450981)}]
    colors += [{"name": "guo", "rgb": (0.988, 0.141, 0.757)}]
    colors += [{"name": "unia", "rgb": (0.678, 0.0, 0.486)}]
    colors += [{"name": "fai", "rgb": (0.282, 0.576, 0.141)}]

    # define colors as list of dictionaries with "name":str, "rgb":tuple entries
    if "colors" in kwargs:
        colors += kwargs["colors"]
        kwargs.pop("colors")
    for color in colors:
        result += "COLOR {} {} {} {}\n".format(color["name"], *tuple(color["rgb"]))

    if group_together is True:
        group_together = "TOUCH"

    qubits = []
    for gate in list_of_gates:
        if isinstance(gate, GenericGate):
            if gate.name in ["initialstate", "simple", "trotter"]:
                qubits.extend(gate.qubits)
        elif isinstance(gate, Double_excitation):
            qubits.append(gate.i // 2)
            qubits.append(gate.j // 2)
            qubits.append(gate.k // 2)
            qubits.append(gate.l // 2)
        elif isinstance(gate, Single_excitation):
            qubits.append(gate.i // 2)
            qubits.append(gate.j // 2)
        elif isinstance(gate, RenderableGate):
            qubits.extend(gate.used_wires())
        else:
            qubits.append(gate.i)
            qubits.append(gate.j)
    qubits = list(set(qubits))

    # define wires
    names = dict()
    if qubit_names is None:
        qubit_names = qubits
    if "wire_colors" in kwargs:
        wcolors = kwargs["wire_colors"]
        kwargs.pop("wire_colors")
    else:
        wcolors = {}
    for i, q in enumerate(qubits):
        name = "a" + str(q)
        if qubit_names[i] in wcolors:
            color = wcolors[qubit_names[i]]
        else:
            color = "black"
        names[q] = name
        result += f"color={color} " + name + " W " + str(qubit_names[i]) + "\n"
    for i, q in enumerate(qubits):
        result += names[q] + " /\n"
    spin = {0: -3, 1: 3}
    for g in list_of_gates:
        param = None
        if isinstance(g, OrbitalRotatorGate):
            shape = 2
            tcol = textcolor1
            gcol = gatecolor1
        elif isinstance(g, Single_excitation):
            shape1 = spin[g.i % 2]
            shape2 = spin[g.j % 2]
            tcol = textcolor1
            gcol = gatecolor1
        elif isinstance(g, Double_excitation):
            if not g.type:
                shape = 6
            elif g.type == 1:
                shape1 = 6
                shape2 = spin[g.j % 2]
                shape3 = spin[g.l % 2]
            elif g.type == 2:
                shape3 = 6
                shape1 = spin[g.i % 2]
                shape2 = spin[g.k % 2]
            else:
                shape1 = spin[g.i % 2]
                shape2 = spin[g.j % 2]
                shape3 = spin[g.k % 2]
                shape4 = spin[g.l % 2]
            tcol = textcolor2
            gcol = gatecolor2
        else:
            shape = 6
            tcol = textcolor2
            gcol = gatecolor2
        if hasattr(g, "angle"):
            if not isinstance(g.angle, numbers.Number) and mark_parametrized_gates:
                tcol = textcolor3
                gcol = gatecolor3
            if isinstance(g.angle, numbers.Number) and color_range:
                if g.unit_of_pi:
                    angle = int(((g.angle % 2) / 2) * 100)
                else:
                    angle = int(abs((g.angle / (2 * pi)) * 100)) % 100
                gcol = f"{color_from}!{angle}!{color_to}"
            param = g.angle

        if isinstance(param, numbers.Number):
            param = "{:1.2f}".format(param)
        if isinstance(g, GenericGate):
            if g.name != 'trotter':
                for q in g.qubits:
                    result += " a{qubit} ".format(qubit=q)
            if g.name == "initialstate":
                result += " G I "
            if g.name == "simple":
                result += " G G "
            if g.name == "single":
                result += " color=blue "
            if g.name == "double":
                result += " color=fai "
            if g.name == "trotter":
                text = tq.circuit.qpic.export_to_qpic(
                    circuit=tq.gates.Trotterized(generator=g.U.generator, angle=g.U._parameter))
                while text[0:len("COLOR")] == "COLOR":
                    text = text[text.find("\n") + 1:]
                while text[0] == "a":
                    text = text[text.find("\n") + 1:]
                result += text
        elif isinstance(g, Single_excitation):
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=g.i // 2,
                                                                                                shape=shape1, gcol=gcol,
                                                                                                tcol="{" + tcol + "}",
                                                                                                op="")
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=g.j // 2,
                                                                                                shape=shape2, gcol=gcol,
                                                                                                tcol="{" + tcol + "}",
                                                                                                op="")
        elif isinstance(g, Double_excitation):
            if not g.type:
                result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=g.i // 2,
                                                                                                    shape=shape,
                                                                                                    gcol=gcol,
                                                                                                    tcol="{" + tcol + "}",
                                                                                                    op="")
                result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=g.j // 2,
                                                                                                    shape=shape,
                                                                                                    gcol=gcol,
                                                                                                    tcol="{" + tcol + "}",
                                                                                                    op="")
            elif g.type == 1:
                result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=g.i // 2,
                                                                                                    shape=shape1,
                                                                                                    gcol=gcol,
                                                                                                    tcol="{" + tcol + "}",
                                                                                                    op="")
                result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=g.j // 2,
                                                                                                    shape=shape2,
                                                                                                    gcol=gcol,
                                                                                                    tcol="{" + tcol + "}",
                                                                                                    op="")
                result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=g.l // 2,
                                                                                                    shape=shape3,
                                                                                                    gcol=gcol,
                                                                                                    tcol="{" + tcol + "}",
                                                                                                    op="")
            elif g.type == 2:
                result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=g.i // 2,
                                                                                                    shape=shape1,
                                                                                                    gcol=gcol,
                                                                                                    tcol="{" + tcol + "}",
                                                                                                    op="")
                result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=g.k // 2,
                                                                                                    shape=shape2,
                                                                                                    gcol=gcol,
                                                                                                    tcol="{" + tcol + "}",
                                                                                                    op="")
                result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=g.j // 2,
                                                                                                    shape=shape3,
                                                                                                    gcol=gcol,
                                                                                                    tcol="{" + tcol + "}",
                                                                                                    op="")
            else:
                result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=g.i // 2,
                                                                                                    shape=shape1,
                                                                                                    gcol=gcol,
                                                                                                    tcol="{" + tcol + "}",
                                                                                                    op="")
                result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=g.j // 2,
                                                                                                    shape=shape2,
                                                                                                    gcol=gcol,
                                                                                                    tcol="{" + tcol + "}",
                                                                                                    op="")
                result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=g.k // 2,
                                                                                                    shape=shape3,
                                                                                                    gcol=gcol,
                                                                                                    tcol="{" + tcol + "}",
                                                                                                    op="")
                result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=g.l // 2,
                                                                                                    shape=shape4,
                                                                                                    gcol=gcol,
                                                                                                    tcol="{" + tcol + "}",
                                                                                                    op="")
        elif isinstance(g, RenderableGate):
            result += g.render_circuit() + " "
        else:
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=g.i, shape=shape,
                                                                                                gcol=gcol,
                                                                                                tcol="{" + tcol + "}",
                                                                                                op="")
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=g.j, shape=shape,
                                                                                                gcol=gcol,
                                                                                                tcol="{" + tcol + "}",
                                                                                                op="")

        result += "\n"
        if hasattr(group_together, "upper"):
            for t in qubits:
                result += "a{} ".format(t)
            result += "{}\n".format(group_together.upper())

    if filename is not None:
        filenamex = filename
        if not filenamex.endswith(".qpic"):
            filenamex = filename + ".qpic"
        if filepath is not None:
            filenamex = "{}/{}".format(filepath, filenamex)
        with open(filenamex, "w") as file:
            file.write(result)
    return result


def qpic_to_pdf(filename, filepath=None):
    if filepath:
        subprocess.call(["qpic", "{}.qpic".format(filename), "-f", "pdf"], cwd=filepath)
    else:
        subprocess.call(["qpic", "{}.qpic".format(filename), "-f", "pdf"])


def qpic_to_png(filename, filepath=None):
    if filepath:
        subprocess.call(["qpic", "{}.qpic".format(filename), "-f", "png"], cwd=filepath)
    else:
        subprocess.call(["qpic", "{}.qpic".format(filename), "-f", "png"])


def from_circuit(U, n_qubits_is_double: bool = False, *args, **kwargs):
    circuit = U._gates
    res = []
    was_ur = False
    for i, gate in enumerate(circuit):
        if gate._name == 'FermionicExcitation':
            index = ()
            for pair in gate.indices:
                for so in pair:
                    index += (so,)
            if len(index) == 2:
                if was_ur:
                    was_ur = False
                    continue
                elif gate != circuit[-1] and circuit[i + 1]._name == 'FermionicExcitation' and len(
                        circuit[i + 1].indices[0]) == 2 and circuit[i + 1].indices[0][0] // 2 == index[0] // 2 and \
                        circuit[i + 1].indices[0][1] // 2 == index[1] // 2:  # hope you enjoy this conditional
                    res.append(OrbitalRotatorGate(index[0] // 2, index[1] // 2, angle=gate._parameter, *args, **kwargs))
                    was_ur = True
                else:
                    res.append(Single_excitation(index[0], index[1], angle=gate._parameter, *args, **kwargs))
            else:
                if index[0] // 2 == index[2] // 2 and index[1] // 2 == index[
                    3] // 2:  ## TODO: Maybe generalized for further excitations
                    res.append(OrbitalRotatorGate(index[0] // 2, index[1] // 2, angle=gate._parameter, *args, **kwargs))
                else:
                    res.append(Double_excitation(index[0], index[1], index[2], index[3], angle=gate._parameter, *args,
                                                 **kwargs))
        elif gate._name == 'QubitExcitation':
            index = gate._target
            if len(index) == 2:
                if was_ur:
                    was_ur = False
                    continue
                elif gate != circuit[-1] and circuit[i + 1]._name == 'QubitExcitation' and len(
                        circuit[i + 1]._target) == 2 and circuit[i + 1]._target[0] // 2 == index[0] // 2 and \
                        circuit[i + 1]._target[1] // 2 == index[1] // 2:  # hope you enjoy this conditional
                    res.append(OrbitalRotatorGate(index[0] // 2, index[1] // 2, angle=gate._parameter, *args, **kwargs))
                    was_ur = False
                else:
                    res.append(Single_excitation(index[0], index[1], angle=gate._parameter, *args, **kwargs))
            else:
                if index[0] // 2 == index[2] // 2 and index[1] // 2 == index[
                    3] // 2:  ## TODO: Maybe generalized for further excitations
                    res.append(OrbitalRotatorGate(index[0] // 2, index[1] // 2, angle=gate._parameter, *args, **kwargs))
                else:
                    res.append(Double_excitation(index[0], index[1], index[2], index[3], angle=gate._parameter, *args,
                                                 **kwargs))
        else:
            res.append(GenericGate(U=gate, name="trotter", n_qubits_is_double=n_qubits_is_double, *args, **kwargs))
    return res


if __name__ == "__main__":
    circuit = [
        GenericGate(U=tq.gates.X([0, 1, 2]), name="initialstate"),
        PairCorrelatorGate(0, 1, 1.0),
        PairCorrelatorGate(5, 4, 1.0),
        PairCorrelatorGate(3, 2, "a"),
        GenericGate(U=tq.gates.X([0, 3, 5]), name="simple"),
        OrbitalRotatorGate(0, 3, 1.0),
        OrbitalRotatorGate(5, 4, 1.0),
        OrbitalRotatorGate(0, 5, "a"),
    ]

    # circuit = [
    #     OrbitalRotatorGate(0,1,1.0),
    #     PairCorrelatorGate(0,1,1.0)
    # ]

    # circuit = [
    #     InitialState(U=tq.gates.X([0,1])+tq.gates.CNOT(0,3)),
    #     OrbitalRotatorGate(1,2,"a")
    # ]

    filename = "test_filename"
    cwd = os.getcwd()

    # export_to_qpic(list_of_gates=circuit, gatecolor1="tq", gatecolor2="tq", mark_parametrized_gates=True, filename=filename) # different colors
    # export_to_qpic(list_of_gates=circuit, filename=filename)
    # qpic_to_png(filename, cwd)
