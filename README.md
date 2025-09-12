# State Specfic Measurement Protocols for the Variational Quantum Eigensolver (Prototype Implementation)
The repository is designed to run the general procedure and show some example data from: [arxiv:2504.03019](https://arxiv.org/abs/2504.03019)

The core methods `fold_rotators` and `get_hcb_part` (to rotate the integrals and extract the HCB Hamiltonian) are defined in the file `measurements_utils.py`. The additional utils files are convenience files made to create the quantum circuit (for Scenario II). The circuits can be produced through the [Tequila](https://github.com/tequilahub/tequila) library. Check documentation and tutorials for more.

Randomized geometries have been generated with [quanti-gin](https://github.com/nylser/quanti-gin) and circuits depictions are made with [qpic](https://github.com/qpic/qpic). Read online documentation for more.


# Installation
The following will work on OSX and Linux (no PySCF on Windows)

Install this program with all the dependencies like this:

```bash
conda create -n myenv python=3.10
conda activate myenv

cd mvb
pip install -e .
```

# Usage
In the example folder you can find two examples for Scenario I, two examples for Scenario II as well as the computations for the BeH2 molecule and excited states of the H4 molecule.