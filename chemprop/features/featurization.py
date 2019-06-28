from argparse import Namespace
from copy import deepcopy
from typing import List, Set, Tuple, Union

import numpy as np
from rdkit import Chem
import torch

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14

# Memoization
SMILES_TO_GRAPH = {}


def clear_cache():
    """Clears featurization cache."""
    global SMILES_TO_GRAPH
    SMILES_TO_GRAPH = {}


def get_atom_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM


def get_bond_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


class MolGraph:
    """A MolGraph represents the graph structure and featurization of a single molecule."""
    def __init__(self,
                 n_atoms: int,
                 n_bonds: int,
                 f_atoms: List[List[Union[bool, int, float]]],
                 f_bonds: List[List[Union[bool, int, float]]],
                 a2b: List[List[int]],
                 b2a: List[int],
                 b2revb: List[int]):
        """
        Creates a MolGraph.

        :param n_atoms: Number of atoms.
        :param n_bonds: Number of bonds.
        :param f_atoms: Mapping from atom index to atom features.
        :param f_bonds: Mapping from bond index to concat(in_atom, bond) features.
        :param a2b: Mapping from atom index to incoming bond indices.
        :param b2a: Mapping from bond index to the index of the atom the bond is coming from.
        :param b2revb: Mapping from bond index to the index of the reverse bond.
        """
        self.n_atoms = n_atoms
        self.n_bonds = n_bonds
        self.f_atoms = f_atoms
        self.f_bonds = f_bonds
        self.a2b = a2b
        self.b2a = b2a
        self.b2revb = b2revb

    @classmethod
    def from_smiles(cls, smiles: str, args: Namespace):
        """
        Computes the graph structure and featurization of a molecule and creates a MolGraph.

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        n_atoms = 0  # number of atoms
        n_bonds = 0  # number of bonds
        f_atoms = []  # mapping from atom index to atom features
        f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        a2b = []  # mapping from atom index to incoming bond indices
        b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = []  # mapping from bond index to the index of the reverse bond

        # Convert smiles to molecule
        mol = Chem.MolFromSmiles(smiles)

        # fake the number of "atoms" if we are collapsing substructures
        n_atoms = mol.GetNumAtoms()

        # Get atom features
        for i, atom in enumerate(mol.GetAtoms()):
            f_atoms.append(atom_features(atom))
        f_atoms = [f_atoms[i] for i in range(n_atoms)]

        for _ in range(n_atoms):
            a2b.append([])

        # Get bond features
        for a1 in range(n_atoms):
            for a2 in range(a1 + 1, n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = bond_features(bond)

                if args.atom_messages:
                    f_bonds.append(f_bond)
                    f_bonds.append(f_bond)
                else:
                    f_bonds.append(f_atoms[a1] + f_bond)
                    f_bonds.append(f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = n_bonds
                b2 = b1 + 1
                a2b[a2].append(b1)  # b1 = a1 --> a2
                b2a.append(a1)
                a2b[a1].append(b2)  # b2 = a2 --> a1
                b2a.append(a2)
                b2revb.append(b2)
                b2revb.append(b1)
                n_bonds += 2

        return MolGraph(
            n_atoms=n_atoms,
            n_bonds=n_bonds,
            f_atoms=f_atoms,
            f_bonds=f_bonds,
            a2b=a2b,
            b2a=b2a,
            b2revb=b2revb
        )


class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph], args: Namespace):
        self.atom_fdim = get_atom_fdim(args)
        self.bond_fdim = get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def extract_subgraph(atoms: Set[int],
                     bond_to_atom_pair: List[Tuple[int, int]],
                     mol_graph: MolGraph) -> MolGraph:
    """
    Extracts the subgraph of a MolGraph containing the provided atoms.
    
    :param atoms: A set of atom indices which will be used to extract a subgraph.
    :param bond_to_atom_pair: A mapping from bond indices to pairs of atom indices (a1 --> a2).
    :param mol_graph: A MolGraph for a molecule.
    :return: A subgraph MolGraph containing all the provided atom indices.
    """
    # Determine bonds in subgraph
    bonds = {b for b, (a1, a2) in enumerate(bond_to_atom_pair) if a1 in atoms and a2 in atoms}

    # Convert atoms and bonds to sorted lists
    atom_list, bond_list = sorted(atoms), sorted(bonds)

    # Determine mapping from old indices to new indices
    a2a = {old_a: new_a for new_a, old_a in enumerate(atom_list)}
    b2b = {old_b: new_b for new_b, old_b in enumerate(bond_list)}

    # Extract sub-tensors
    subgraph = MolGraph(
        n_atoms=len(atoms),
        n_bonds=len(bonds),
        f_atoms=[mol_graph.f_atoms[a] for a in atom_list],
        f_bonds=[mol_graph.f_bonds[b] for b in bond_list],
        a2b=[[b2b[b] for b in mol_graph.a2b[a] if b in bonds] for a in atom_list],
        b2a=[a2a[mol_graph.b2a[b]] for b in bond_list],
        b2revb=[b2b[mol_graph.b2revb[b]] for b in bond_list]
    )

    return subgraph


def extract_substructure_and_context_subgraphs(smiles: str,
                                               mol_graph: MolGraph,
                                               args: Namespace) -> List[Tuple[MolGraph, MolGraph]]:
    """
    Extracts substructure and context subgraphs for pretraining.

    :param smiles: SMILES string for the molecule.
    :param mol_graph: A MolGraph containing the features and connectivity of the molecule.
    :param args: Arguments.
    :return: A list of of tuples of (substructure, context) subgraphs for each atom on the molecule.
    """
    subgraphs = []

    # Get distances between atoms
    mol = Chem.MolFromSmiles(smiles)
    distances = Chem.GetDistanceMatrix(mol)  # num_atoms x num_atoms

    # Get mapping from bond index to atoms the bond connects
    bond_to_atom_pair = []
    for a1 in range(mol.GetNumAtoms()):
        for a2 in range(a1 + 1, mol.GetNumAtoms()):
            bond = mol.GetBondBetweenAtoms(a1, a2)

            if bond is None:
                continue

            bond_to_atom_pair.append((a1, a2))  # a1 --> a2
            bond_to_atom_pair.append((a2, a1))  # a2 --> a1

    # Note: Assumes MolGraph indexed the atoms and bonds in the same way
    for a in range(mol.GetNumAtoms()):
        # Substructure subgraph (distance <= r1)
        substructure_atoms = set(np.where(distances[a] <= args.inner_context_radius)[0])
        substructure_graph = extract_subgraph(substructure_atoms, bond_to_atom_pair, mol_graph)

        # Context subgraph (r1 <= distance <= r2)
        context_atoms = set(np.where(np.logical_and(distances[a] >= args.inner_context_radius, distances[a] <= args.outer_context_radius))[0])
        context_graph = extract_subgraph(context_atoms, bond_to_atom_pair, mol_graph)

        # Add subgraphs
        subgraphs.append((substructure_graph, context_graph))

    return subgraphs


def mol2graph(smiles_batch: List[str],
              args: Namespace) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    all_mol_graphs = []

    for i, smiles in enumerate(smiles_batch):
        if smiles in SMILES_TO_GRAPH:
            mol_graphs = SMILES_TO_GRAPH[smiles]
        else:
            mol_graph = MolGraph.from_smiles(smiles, args)

            # Context prediction subgraph extraction for node-level pretraining
            if args.dataset_type == 'pretraining':
                subgraphs = extract_substructure_and_context_subgraphs(smiles, mol_graph, args)
                mol_graphs = list(sum(subgraphs, tuple()))  # [substructure_1, context_1, substructure_2, context_2, ...]
            else:
                mol_graphs = [mol_graph]

            if not args.no_cache:
                SMILES_TO_GRAPH[smiles] = mol_graphs

        all_mol_graphs += mol_graphs

    return BatchMolGraph(all_mol_graphs, args)
