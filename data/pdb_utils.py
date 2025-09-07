# Source https://github.com/THUNLP-MT/GET

from copy import copy, deepcopy
import math
import os
from typing import Dict, List, Tuple, Optional

import numpy as np

from .tokenizer.tokenize_3d import TOKENIZER


BACKBONE = ['N', 'CA', 'C', 'O']

SIDECHAIN = {
    'G': [],   # -H
    'A': ['CB'],  # -CH3
    'V': ['CB', 'CG1', 'CG2'],  # -CH-(CH3)2
    'L': ['CB', 'CG', 'CD1', 'CD2'],  # -CH2-CH(CH3)2
    'I': ['CB', 'CG1', 'CG2', 'CD1'], # -CH(CH3)-CH2-CH3
    'F': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],  # -CH2-C6H5
    'W': ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],  # -CH2-C8NH6
    'Y': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],  # -CH2-C6H4-OH
    'D': ['CB', 'CG', 'OD1', 'OD2'],  # -CH2-COOH
    'H': ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],  # -CH2-C3H3N2
    'N': ['CB', 'CG', 'OD1', 'ND2'],  # -CH2-CONH2
    'E': ['CB', 'CG', 'CD', 'OE1', 'OE2'],  # -(CH2)2-COOH
    'K': ['CB', 'CG', 'CD', 'CE', 'NZ'],  # -(CH2)4-NH2
    'Q': ['CB', 'CG', 'CD', 'OE1', 'NE2'],  # -(CH2)-CONH2
    'M': ['CB', 'CG', 'SD', 'CE'],  # -(CH2)2-S-CH3
    'R': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],  # -(CH2)3-NHC(NH)NH2
    'S': ['CB', 'OG'],  # -CH2-OH
    'T': ['CB', 'OG1', 'CG2'],  # -CH(CH3)-OH
    'C': ['CB', 'SG'],  # -CH2-SH
    'P': ['CB', 'CG', 'CD'],  # -C3H6
}

ATOMS = [ # Periodic Table
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og'
]


def format_atom_element(atom: str):
    atom = atom.upper()
    if len(atom) == 2:
        atom = atom[0] + atom[1].lower()
    return atom


class Vocab:

    def __init__(self):
        self.PAD, self.MASK, self.UNK = '#', '*', '?'
        self.GLB = '&'  # global node
        specials = [# special added
                (self.PAD, 'PAD'), (self.MASK, 'MASK'), (self.UNK, 'UNK'), # pad / mask / unk
                (self.GLB, '<G>')  # global node
            ]
        # specials = []
        # (symbol, abbrv)
        aas = [  # amino acids
                ('G', 'GLY'), ('A', 'ALA'), ('V', 'VAL'), ('L', 'LEU'),
                ('I', 'ILE'), ('F', 'PHE'), ('W', 'TRP'), ('Y', 'TYR'),
                ('D', 'ASP'), ('H', 'HIS'), ('N', 'ASN'), ('E', 'GLU'),
                ('K', 'LYS'), ('Q', 'GLN'), ('M', 'MET'), ('R', 'ARG'),
                ('S', 'SER'), ('T', 'THR'), ('C', 'CYS'), ('P', 'PRO') # 20 aa
                # ('U', 'SEC') # 21 aa for eukaryote
            ]

        bases = [ # bases for RNA/DNA
                ('DA', 'DA'), ('DG', 'DG'), ('DC', 'DC'), ('DT', 'DT'), # DNA
                ('RA', 'RA'), ('RG', 'RG'), ('RC', 'RC'), ('RU', 'RU'), # RNA, RI = inosine
        ]

        sms = [(atom.lower(), atom) for atom in ATOMS]
        
        frags = [] # principal subgraphs
        if len(TOKENIZER):
            _tmp_map = { atom: True for atom in ATOMS }
            for i, smi in enumerate(TOKENIZER.get_frag_smiles()):
                if smi in _tmp_map: # single atom
                    continue
                frags.append((str(i), smi))
        
        self.aas, self.bases, self.sms, self.frags = aas, bases, sms, frags

        self.atom_pad, self.atom_mask, self.atom_global = 'p', 'm', 'g'
        self.atom_pos_pad, self.atom_pos_mask, self.atom_pos_global = 'p', 'm', 'g'
        self.atom_pos_sm = 'sm'  # small molecule


        _all = specials + aas + bases + sms + frags
        self.symbol2idx, self.abrv2idx = {}, {}
        self.idx2symbol, self.idx2abrv = [], []
        for i, (symbol, abrv) in enumerate(_all):
            self.symbol2idx[symbol] = i
            self.abrv2idx[abrv] = i
            self.idx2symbol.append(symbol)
            self.idx2abrv.append(abrv)
        self.special_mask = [1 for _ in specials] + [0 for _ in aas + bases + sms + frags]
        assert len(self.symbol2idx) == len(self.idx2symbol)
        assert len(self.abrv2idx) == len(self.idx2abrv)
        assert len(self.idx2symbol) == len(self.idx2abrv)

        # atom level vocab
        self.idx2atom = [self.atom_pad, self.atom_mask, self.atom_global] + ATOMS
        self.idx2atom_pos = [self.atom_pos_pad, self.atom_pos_mask, self.atom_pos_global, ''] + \
                            ['A', 'B', 'G', 'D', 'E', 'Z', 'H', 'XT', 'P'] + \
                            [self.atom_pos_sm] + ["'"]  # SM is for atoms in small molecule, 'P' for O1P, O2P, O3P, "'" for bases
        self.atom2idx, self.atom_pos2idx = {}, {}
        for i, atom in enumerate(self.idx2atom):
            self.atom2idx[atom] = i
        for i, atom_pos in enumerate(self.idx2atom_pos):
            self.atom_pos2idx[atom_pos] = i
    
    def load_tokenizer(self, method: Optional[str]):
        if method is None:
            return
        TOKENIZER.load(method)
        self.__init__()

    def abrv_to_symbol(self, abrv):
        idx = self.abrv_to_idx(abrv)
        return None if idx is None else self.idx_to_symbol(idx)

    def symbol_to_abrv(self, symbol):
        idx = self.symbol_to_idx(symbol)
        return None if idx is None else self.idx2abrv[idx]

    def abrv_to_idx(self, abrv):
        return self.abrv2idx.get(abrv, self.abrv2idx['UNK'])

    def symbol_to_idx(self, symbol):
        return self.symbol2idx.get(symbol, self.abrv2idx['UNK'])
    
    def idx_to_symbol(self, idx):
        return self.idx2symbol[idx]

    def idx_to_abrv(self, idx):
        return self.idx2abrv[idx]

    def get_pad_idx(self):
        return self.symbol_to_idx(self.PAD)

    def get_mask_idx(self):
        return self.symbol_to_idx(self.MASK)
    
    def get_special_mask(self):
        return copy(self.special_mask)

    def get_atom_pad_idx(self):
        return self.atom2idx[self.atom_pad]
    
    def get_atom_mask_idx(self):
        return self.atom2idx[self.atom_mask]
    
    def get_atom_global_idx(self):
        return self.atom2idx[self.atom_global]
    
    def get_atom_pos_pad_idx(self):
        return self.atom_pos2idx[self.atom_pos_pad]

    def get_atom_pos_mask_idx(self):
        return self.atom_pos2idx[self.atom_pos_mask]
    
    def get_atom_pos_global_idx(self):
        return self.atom_pos2idx[self.atom_pos_global]
    
    def idx_to_atom(self, idx):
        return self.idx2atom[idx]

    def atom_to_idx(self, atom):
        return self.atom2idx.get(atom, self.atom2idx[self.atom_mask])

    def idx_to_atom_pos(self, idx):
        return self.idx2atom_pos[idx]
    
    def atom_pos_to_idx(self, atom_pos):
        return self.atom_pos2idx.get(atom_pos, self.atom_pos2idx[self.atom_pos_mask])

    def get_num_atom_type(self):
        return len(self.idx2atom)
    
    def get_num_atom_pos(self):
        return len(self.idx2atom_pos)

    def get_num_amino_acid_type(self):
        return len(self.special_mask) - sum(self.special_mask)

    def __len__(self):
        return len(self.symbol2idx)


VOCAB = Vocab()


def format_aa_abrv(abrv):  # special cases
    if abrv == 'MSE':
        return 'MET' # substitue MSE with MET
    return abrv


class Atom:
    def __init__(self, atom_name: str, coordinate: List, element: str, pos_code: str=None):
        self.name = atom_name
        self.coordinate = coordinate
        self.element = format_atom_element(element)
        if pos_code is None:
            pos_code = atom_name.lstrip(element)
            pos_code = ''.join((c for c in pos_code if not c.isdigit()))
            self.pos_code = pos_code
        else:
            self.pos_code = pos_code

    def get_element(self):
        return self.element
    
    def get_coord(self):
        return copy(self.coordinate)
    
    def get_pos_code(self):
        return self.pos_code
    
    def __str__(self) -> str:
        return self.name


def dist_matrix_from_coords(coords1, mask1, coords2, mask2):
    """
    Calculate distance matrix between two sets of coordinates
    
    Args:
        coords1: First set of coordinates [N1, 3]
        mask1: Mask for first set [N1]
        coords2: Second set of coordinates [N2, 3] 
        mask2: Mask for second set [N2]
        
    Returns:
        Distance matrix [N1, N2]
    """
    # Calculate pairwise distances
    coords1 = np.expand_dims(coords1, axis=1)  # [N1, 1, 3]
    coords2 = np.expand_dims(coords2, axis=0)  # [1, N2, 3]
    
    diff = coords1 - coords2  # [N1, N2, 3]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))  # [N1, N2]
    
    # Apply masks
    mask1 = np.expand_dims(mask1, axis=1)  # [N1, 1]
    mask2 = np.expand_dims(mask2, axis=0)  # [1, N2]
    combined_mask = mask1 * mask2  # [N1, N2]
    
    # Set masked distances to a large value
    dist = np.where(combined_mask, dist, np.inf)
    
    return dist