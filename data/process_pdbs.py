import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import os
import sys
import traceback
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.converter.pdb_lig_to_blocks import extract_pdb_ligand
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.dataset import blocks_interface, blocks_to_data

def parse_args():
    parser = argparse.ArgumentParser(description='Process PDB data for embedding with ATOMICA - Combined ligand and pocket processing')
    parser.add_argument('--data_index_file', type=str, default='LP-PDBbind/2016/test_combined.csv')
    parser.add_argument('--out_path', type=str, default='', help='Output path')
    parser.add_argument('--fragmentation_method', type=str, default='PS_300', choices=['PS_300'], help='fragmentation method for small molecule pockets')
    return parser.parse_args()

def process_PL_pdb(input_type,
                   pdb_file,
                   pdb_id,
                   rec_chain,
                   lig_code,
                   lig_chain,
                   smiles,
                   lig_resi,
                   fragmentation_method=None):
    items = []

    if input_type == 'ligand':

        list_lig_blocks, list_lig_indexes = extract_pdb_ligand(
            pdb_file,
            lig_code,
            lig_chain,
            smiles,
            lig_idx=lig_resi,
            use_model=0,
            fragmentation_method=fragmentation_method
        )


        for idx, (lig_blocks, lig_indexes) in enumerate(
                zip(list_lig_blocks, list_lig_indexes)):

            data = blocks_to_data(lig_blocks)

            _id = f"{pdb_id}_{lig_chain}_{lig_code}"
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
            })

    elif input_type == 'pocket':

        rec_blocks, rec_indexes = pdb_to_list_blocks(
            pdb_file,
            selected_chains=rec_chain,
            return_indexes=True
        )

        rec_blocks_flat = sum(rec_blocks, [])
        rec_indexes_flat = sum(rec_indexes, [])

        data = blocks_to_data(rec_blocks_flat)

        _id = f"{pdb_id}_{''.join(rec_chain)}"

        block_to_pdb = {
            blk_idx + 1: pdb_idx
            for blk_idx, pdb_idx in enumerate(rec_indexes_flat)
        }
        items.append({
            'data': data,
            'block_to_pdb_indexes': block_to_pdb,
            'id': _id,
        })
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    return items

def group_chains(list_chain_blocks, list_chain_pdb_indexes, group1, group2):
    group1_chains = []
    group2_chains = []
    group1_indexes = []
    group2_indexes = []
    for chain_blocks, chain_pdb_indexes in zip(list_chain_blocks, list_chain_pdb_indexes):
        if chain_pdb_indexes[0].split("_")[0] in group1:
            group1_chains.extend(chain_blocks)
            group1_indexes.extend(chain_pdb_indexes)
        elif chain_pdb_indexes[0].split("_")[0] in group2:
            group2_chains.extend(chain_blocks)
            group2_indexes.extend(chain_pdb_indexes)
    return [group1_chains, group2_chains], [group1_indexes, group2_indexes]

def process_pdb(pdb_file, pdb_id, group1_chains, group2_chains, dist_th):
    blocks, pdb_indexes = pdb_to_list_blocks(pdb_file, selected_chains=group1_chains+group2_chains, return_indexes=True, use_model=0)
    if len(blocks) != 2:
        blocks, pdb_indexes = group_chains(blocks, pdb_indexes, group1_chains, group2_chains)
    blocks1, blocks2, block1_indexes, block2_indexes = blocks_interface(blocks[0], blocks[1], dist_th, return_indexes=True)
    if len(blocks1) == 0 or len(blocks2) == 0:
        return None
    pdb_indexes_map = {}
    pdb_indexes_map.update(dict(zip(range(1,len(blocks1)+1), [pdb_indexes[0][i] for i in block1_indexes])))# map block index to pdb index, +1 for global block)
    pdb_indexes_map.update(dict(zip(range(len(blocks1)+2,len(blocks1)+len(blocks2)+2), [pdb_indexes[1][i] for i in block2_indexes])))# map block index to pdb index, +1 for global block)
    data = blocks_to_data(blocks1, blocks2)
    return {
        "data": data,
        "id": f"{pdb_id}_{''.join(group1_chains)}_{''.join(group2_chains)}",
        "block_to_pdb_indexes": pdb_indexes_map,
    }

def process_single_row(row, fragmentation_method):
    """Process a single row of the dataframe"""
    try:
        input_type = row['input_type']
        pdb_file = row['pdb_file']
        pdb_id = row['pdb_id']
        chain1 = row['chain1']
        chain2 = row['chain2']
        lig_code = row['lig_code']
        smiles = row['smiles']
        lig_resi = int(row['lig_resi']) if not pd.isna(row['lig_resi']) else None
        chain1 = chain1.split("_") if not pd.isna(row['chain1']) else None
        chain2 = chain2.split("_")[0] if not pd.isna(row['chain2']) else None

        pl_items = process_PL_pdb(
            input_type=input_type,
            pdb_file=pdb_file,
            pdb_id=pdb_id,
            rec_chain=chain1,
            lig_code=lig_code,
            lig_chain=chain2,
            smiles=smiles,
            lig_resi=lig_resi,
            fragmentation_method=fragmentation_method
        )
        return pl_items, None
    except Exception as e:
        print(f"Error processing row {row.name}: {str(e)}")
        traceback.print_exc()
        return [], str(e)

def merge_ligand_pocket_features(ligand_item, pocket_item):
    """
    Merge ligand and pocket features into a single combined item
    Based on merge_data.py logic
    """
    # Verify that base IDs match
    ligand_id = ligand_item.get('id', '').split('_')[0]
    pocket_id = pocket_item.get('id', '').split('_')[0]
    if ligand_id != pocket_id:
        raise ValueError(f"Mismatched IDs: {ligand_item.get('id')} vs {pocket_item.get('id')}")
    
    ld = ligand_item['data']
    pd = pocket_item['data']
    
    # Concatenate features: X, B, A
    combined_data = {
        'X': np.concatenate([pd['X'], ld['X']], axis=0),
        'B': np.concatenate([pd['B'], ld['B']], axis=0),
        'A': np.concatenate([pd['A'], ld['A']], axis=0),
        'block_lengths': pd['block_lengths'] + ld['block_lengths'],
        'segment_ids': [0] * len(pd['block_lengths']) + [1] * len(ld['block_lengths']),
    }
    
    return {
        'id': ligand_id,
        'data': combined_data,
        'block_to_pdb_indexes': {**pocket_item.get('block_to_pdb_indexes', {}), **ligand_item.get('block_to_pdb_indexes', {})}
    }

def process_combined_row(row, fragmentation_method):
    """
    Process a single row that contains both ligand and pocket information
    """
    try:
        pdb_file = row['pdb_file']
        pdb_id = row['pdb_id']
        
        # Process pocket information
        pocket_chain = row['pocket_chain'].split("_") if not pd.isna(row['pocket_chain']) else None
        
        # Process ligand information
        lig_code = row['lig_code']
        lig_chain = row['lig_chain'].split("_")[0] if not pd.isna(row['lig_chain']) else None
        smiles = row['smiles']
        lig_resi = int(row['lig_resi']) if not pd.isna(row['lig_resi']) else None
        
        # Process pocket
        pocket_items = process_PL_pdb(
            input_type='pocket',
            pdb_file=pdb_file,
            pdb_id=pdb_id,
            rec_chain=pocket_chain,
            lig_code=None,
            lig_chain=None,
            smiles=None,
            lig_resi=None,
            fragmentation_method=fragmentation_method
        )
        
        # Process ligand
        ligand_items = process_PL_pdb(
            input_type='ligand',
            pdb_file=pdb_file,
            pdb_id=pdb_id,
            rec_chain=None,
            lig_code=lig_code,
            lig_chain=lig_chain,
            smiles=smiles,
            lig_resi=lig_resi,
            fragmentation_method=fragmentation_method
        )
        
        if len(pocket_items) == 0 or len(ligand_items) == 0:
            raise ValueError("Failed to process pocket or ligand")
        
        # For simplicity, take the first item from each (could be extended for multiple items)
        pocket_item = pocket_items[0]
        ligand_item = ligand_items[0]
        
        # Merge the features
        combined_item = merge_ligand_pocket_features(ligand_item, pocket_item)
        
        return [combined_item], None
        
    except Exception as e:
        print(f"Error processing combined row {row.name}: {str(e)}")
        traceback.print_exc()
        return [], str(e)

def main(args):
    # 读取数据文件
    data_index_file = pd.read_csv(args.data_index_file)
    total_rows = len(data_index_file)
    
    items = []
    failed_count = 0
    
    print(f"Starting to process {total_rows} rows (combined ligand and pocket processing)...")
    
    # Check if this is the old format (with input_type) or new format (combined)
    if 'input_type' in data_index_file.columns:
        # Old format - use original processing
        print("Using original processing mode (separate ligand/pocket files)")
        for idx, (_, row) in enumerate(tqdm(data_index_file.iterrows(), 
                                           total=total_rows, 
                                           desc="Processing PDB files")):
            pl_items, error = process_single_row(row, args.fragmentation_method)
            
            if not error:
                items.extend(pl_items)
            else:
                failed_count += 1
    else:
        # New format - combined processing
        print("Using combined processing mode (ligand and pocket in same row)")
        for idx, (_, row) in enumerate(tqdm(data_index_file.iterrows(), 
                                           total=total_rows, 
                                           desc="Processing combined PDB files")):
            combined_items, error = process_combined_row(row, args.fragmentation_method)
            
            if not error:
                items.extend(combined_items)
            else:
                failed_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Total rows: {total_rows}")
    print(f"Successfully processed items: {len(items)}")
    print(f"Failed rows: {failed_count}")

    print(f"Saving results to: {args.out_path}")
    with open(args.out_path, 'wb') as f:
        pickle.dump(items, f)
    print("Results saved successfully!")

if __name__ == "__main__":
    args = parse_args()
    main(args)