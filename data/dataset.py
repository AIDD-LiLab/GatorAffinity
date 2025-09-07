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


def open_data_file(data_file):
    """
    Open data file - supports both pickle and compressed JSON formats
    """
    if data_file.endswith(".jsonl.gz"):
        return compressed_jsonl_to_dataset(data_file)
    else:
        with open(data_file, 'rb') as f:
            return pickle.load(f)


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