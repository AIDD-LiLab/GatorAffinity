# Source https://github.com/THUNLP-MT/GET
import random
import os
import pandas as pd
import re
import pickle
import argparse
from tqdm.contrib.concurrent import process_map
from os.path import basename, splitext
from typing import List
from collections import Counter
import gzip
import orjson
import numpy as np
import torch
import biotite.structure as bs
import biotite.structure.io.pdb as pdb

from utils.logger import print_log
from .pdb_utils import Atom, VOCAB, dist_matrix_from_coords


MODALITIES = {"PP":0, "PL":1, "Pion":2, "Ppeptide":3, "PRNA":4, "PDNA":5, "RNAL":6, "CSD":7}



class Block:
    def __init__(self, symbol: str, units: List[Atom]) -> None:
        self.symbol = symbol
        self.units = units

    def __len__(self):
        return len(self.units)
    
    def __iter__(self):
        return iter(self.units)

    @property
    def coords(self):
        return np.mean([unit.get_coord() for unit in self.units], axis=0)

    def to_data(self):
        b = VOCAB.symbol_to_idx(self.symbol)
        x, a, positions = [], [], []
        for atom in self.units:
            a.append(VOCAB.atom_to_idx(atom.get_element()))
            x.append(atom.get_coord())
            positions.append(VOCAB.atom_pos_to_idx(atom.get_pos_code()))
        block_len = len(self)
        return b, a, x, positions, block_len
        



def open_data_file(data_file):
    """
    Open data file - supports both pickle and compressed JSON formats
    """
    if data_file.endswith(".jsonl.gz"):
        return compressed_jsonl_to_dataset(data_file)
    else:
        with open(data_file, 'rb') as f:
            return pickle.load(f)


def blocks_to_data(*blocks_list: List[List[Block]]):
    B, A, X, atom_positions, block_lengths, segment_ids = [], [], [], [], [], []
    for i, blocks in enumerate(blocks_list):
        if len(blocks) == 0:
            continue
        # global node
        cur_B = [VOCAB.symbol_to_idx(VOCAB.GLB)]
        cur_A = [VOCAB.get_atom_global_idx()]
        cur_X = [None]
        cur_atom_positions = [VOCAB.get_atom_pos_global_idx()]
        cur_block_lengths = [1]
        # other nodes
        for block in blocks:
            b, a, x, positions, block_len = block.to_data()
            cur_B.append(b)
            cur_A.extend(a)
            cur_X.extend(x)
            cur_atom_positions.extend(positions)
            cur_block_lengths.append(block_len)
        # update coordinates of the global node to the center
        cur_X[0] = np.mean(cur_X[1:], axis=0).tolist()
        for x_i, x in enumerate(cur_X):
            if isinstance(x, np.ndarray):
                cur_X[x_i] = x.tolist()
        cur_segment_ids = [i for _ in cur_B]
        
        # finish these blocks
        B.extend(cur_B)
        A.extend(cur_A)
        X.extend(cur_X)
        atom_positions.extend(cur_atom_positions)
        block_lengths.extend(cur_block_lengths)
        segment_ids.extend(cur_segment_ids)

    data = {
        'X': X,   # [Natom, 2, 3]
        'B': B,             # [Nb], block (residue) type
        'A': A,             # [Natom]
        'atom_positions': atom_positions,  # [Natom]
        'block_lengths': block_lengths,  # [Nresidue]
        'segment_ids': segment_ids,      # [Nresidue]
    }

    return data


def blocks_interface(blocks1, blocks2, dist_th, return_indexes=False):
    blocks_coord, blocks_mask = blocks_to_coords(blocks1 + blocks2)
    blocks1_coord, blocks1_mask = blocks_coord[:len(blocks1)], blocks_mask[:len(blocks1)]
    blocks2_coord, blocks2_mask = blocks_coord[len(blocks1):], blocks_mask[len(blocks1):]
    dist = dist_matrix_from_coords(blocks1_coord, blocks1_mask, blocks2_coord, blocks2_mask)
    
    on_interface = dist < dist_th
    indexes1 = np.nonzero(on_interface.sum(axis=1) > 0)[0]
    indexes2 = np.nonzero(on_interface.sum(axis=0) > 0)[0]

    blocks1 = [blocks1[i] for i in indexes1]
    blocks2 = [blocks2[i] for i in indexes2]

    if return_indexes:
        return blocks1, blocks2, indexes1, indexes2
    else:
        return blocks1, blocks2

def data_to_blocks(data, fragmentation_method=None):
    if fragmentation_method:
        VOCAB.load_tokenizer(fragmentation_method)
    curr_atom_idx = 0
    list_of_blocks = []
    curr_segment_id = 0
    curr_blocks = []
    for block_idx, block in enumerate(data['B']):
        symbol = VOCAB.idx_to_symbol(block)
        if symbol == VOCAB.GLB:
            curr_atom_idx += data['block_lengths'][block_idx]
            continue
        atom_coords = data['X'][curr_atom_idx:curr_atom_idx+data['block_lengths'][block_idx]]
        atom_positions = data['atom_positions'][curr_atom_idx:curr_atom_idx+data['block_lengths'][block_idx]]
        atoms = []
        for i, atom in enumerate(data['A'][curr_atom_idx:curr_atom_idx+data['block_lengths'][block_idx]]):
            atom_name=VOCAB.idx_to_atom(atom)
            if atom_name == VOCAB.atom_global:
                continue
            element=VOCAB.idx_to_atom(atom)
            coordinate=atom_coords[i]
            pos_code=VOCAB.idx_to_atom_pos(atom_positions[i])
            atoms.append(Atom(atom_name=atom_name, element=element, coordinate=coordinate, pos_code=pos_code))
        curr_atom_idx += data['block_lengths'][block_idx]
        if data['segment_ids'][block_idx] != curr_segment_id:
            list_of_blocks.append(curr_blocks)
            curr_blocks = []
            curr_segment_id = data['segment_ids'][block_idx]
        curr_blocks.append(Block(symbol, atoms))
    list_of_blocks.append(curr_blocks)
    return list_of_blocks


def blocks_to_coords(blocks: List[Block]):
    max_n_unit = 0
    coords, masks = [], []
    for block in blocks:
        coords.append([unit.get_coord() for unit in block.units])
        max_n_unit = max(max_n_unit, len(coords[-1]))
        masks.append([1 for _ in coords[-1]])
    
    for i in range(len(coords)):
        num_pad =  max_n_unit - len(coords[i])
        coords[i] = coords[i] + [[0, 0, 0] for _ in range(num_pad)]
        masks[i] = masks[i] + [0 for _ in range(num_pad)]
    
    return np.array(coords), np.array(masks).astype('bool')  # [N, M, 3], [N, M], M == max_n_unit, in mask 0 is for padding

class PDBBindBenchmark(torch.utils.data.Dataset):
    """Dataset for PDBBind benchmark - protein-ligand binding affinity prediction"""
    def __init__(self, data_file):
        super().__init__()
        self.data = open_data_file(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        an example of the returned data
        {
            'X': [Natom, 3],
            'B': [Nblock],
            'A': [Natom],
            'atom_positions': [Natom],
            'block_lengths': [Natom]
            'segment_ids': [Nblock]
            'label': [1]
        }        
        '''
        data = self.data[idx]
        return data

    @classmethod
    def collate_fn(cls, batch):
        keys = ['X', 'B', 'A', 'block_lengths', 'segment_ids']
        types = [torch.float, torch.long, torch.long, torch.long, torch.long]

        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)
        
        res['label'] = torch.tensor([item['label'] for item in batch], dtype=torch.float)
        lengths = [len(item['B']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        return res


class PDBDataset(torch.utils.data.Dataset):
    """Basic PDB dataset class"""
    def __init__(self, data_file):
        self.data = open_data_file(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item

    @classmethod
    def collate_fn(cls, batch):
        """Basic collate function"""
        batch_data = {}
        
        # Handle tensor data
        for key in ["atom_type", "x", "fragment", "fragment_mask", "edge_index", "batch_atom", "batch_fragment"]:
            if key in batch[0]:
                if key == "edge_index":
                    batch_data[key] = torch.cat([item[key] for item in batch], dim=1)
                else:
                    batch_data[key] = torch.cat([item[key] for item in batch], dim=0)
        
        return batch_data


def compressed_jsonl_to_dataset(input_file):
    """Load dataset from compressed JSON lines format"""
    data = []
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(orjson.loads(line))
    return data