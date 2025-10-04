import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.converter.pdb_lig_to_blocks import extract_pdb_ligand
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.dataset import blocks_interface, blocks_to_data

def parse_args():
    parser = argparse.ArgumentParser(description='Process protein-ligand pairs from CSV and output merged pkl file')
    parser.add_argument('--input_csv', type=str,  default='data/example/example.csv',help='Input CSV file with protein and ligand information')
    parser.add_argument('--output_pkl', type=str, default='data/example/example.pkl', help='Output path for merged pkl file')
    parser.add_argument('--fragmentation_method', type=str, default='PS_300', choices=['PS_300'], help='Fragmentation method for ligands')
    return parser.parse_args()

def process_ligand(pdb_file, pdb_id, lig_code, smiles, lig_resi, fragmentation_method):
    """Process ligand from PDB file and return processed items"""
    items = []

    try:
        list_lig_blocks, list_lig_indexes = extract_pdb_ligand(
            pdb_file,
            lig_code,
            smiles,
            lig_idx=lig_resi,
            use_model=0,
            fragmentation_method=fragmentation_method
        )

        for idx, (lig_blocks, lig_indexes) in enumerate(zip(list_lig_blocks, list_lig_indexes)):
            data = blocks_to_data(lig_blocks)
            _id = f"{pdb_id}_{lig_code}"
            if len(list_lig_blocks) > 1:
                _id = f"{_id}_{idx}"

            block_to_pdb = {
                blk_idx + 1: pdb_idx
                for blk_idx, pdb_idx in enumerate(lig_indexes)
            }

            items.append({
                'data': data,
                'block_to_pdb_indexes': block_to_pdb,
                'id': _id,
                'type': 'ligand'
            })
    except Exception as e:
        print(f"Error processing ligand {pdb_id}: {e}")

    return items

def process_protein(pdb_file, pdb_id, protein_chains):
    """Process protein from PDB file and return processed items"""
    items = []

    try:
        protein_blocks, protein_indexes = pdb_to_list_blocks(
            pdb_file,
            selected_chains=protein_chains,
            return_indexes=True,
            is_rna=False,
        )

        protein_blocks_flat = sum(protein_blocks, [])
        protein_indexes_flat = sum(protein_indexes, [])
        data = blocks_to_data(protein_blocks_flat)
        _id = f"{pdb_id}_{''.join(protein_chains)}"

        block_to_pdb = {
            blk_idx + 1: pdb_idx
            for blk_idx, pdb_idx in enumerate(protein_indexes_flat)
        }

        items.append({
            'data': data,
            'block_to_pdb_indexes': block_to_pdb,
            'id': _id,
            'type': 'protein'
        })
    except Exception as e:
        print(f"Error processing protein {pdb_id}: {e}")

    return items

def merge_protein_ligand(protein_item, ligand_item, label=None):
    """Merge protein and ligand data following merge_data.py logic"""
    pd_data = protein_item['data']
    ld_data = ligand_item['data']

    combined_entry = {
        'id': ligand_item['id'],
        'X': np.concatenate([pd_data['X'], ld_data['X']], axis=0),
        'B': np.concatenate([pd_data['B'], ld_data['B']], axis=0),
        'A': np.concatenate([pd_data['A'], ld_data['A']], axis=0),
        'block_lengths': pd_data['block_lengths'] + ld_data['block_lengths'],
        'segment_ids': [0] * len(pd_data['block_lengths']) + [1] * len(ld_data['block_lengths'])
    }

    if label is not None:
        combined_entry['label'] = label

    return combined_entry

def process_csv_row(row, fragmentation_method):
    """Process a single CSV row containing protein and ligand information"""
    try:
        pdb_id = row['pdb_id']
        protein_pdb = row['protein_pdb']
        ligand_pdb = row['ligand_pdb']
        protein_chains = row['protein_chains'].split('_') if pd.notna(row['protein_chains']) else ['A']
        lig_code = row['lig_code'] if pd.notna(row['lig_code']) else 'UNL'
        smiles = row['smiles'] if pd.notna(row['smiles']) else ''
        lig_resi = int(row['lig_resi']) if pd.notna(row['lig_resi']) else None
        label = float(row['label']) if pd.notna(row['label']) else None

        protein_items = process_protein(protein_pdb, pdb_id, protein_chains)
        ligand_items = process_ligand(ligand_pdb, pdb_id,  lig_code, smiles, lig_resi, fragmentation_method)

        combined_items = []
        for protein_item in protein_items:
            for ligand_item in ligand_items:
                combined_item = merge_protein_ligand(protein_item, ligand_item, label)
                combined_items.append(combined_item)

        return combined_items

    except Exception as e:
        print(f"Error processing row {row.get('pdb_id', 'unknown')}: {e}")
        return []

def main(args):
    df = pd.read_csv(args.input_csv)
    all_items = []

    print(f"Processing {len(df)} rows from {args.input_csv}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        items = process_csv_row(row, args.fragmentation_method)
        all_items.extend(items)

    output_dir = os.path.dirname(args.output_pkl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(all_items, f)

    print(f"Processed {len(all_items)} protein-ligand pairs, saved to {args.output_pkl}")

if __name__ == "__main__":
    main(parse_args())