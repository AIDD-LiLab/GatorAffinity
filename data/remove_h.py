#!/usr/bin/env python
"""
Simple script to remove hydrogen atoms from PDB files
Usage: python remove_h.py input.pdb output.pdb
"""

import sys
from rdkit import Chem

def remove_hydrogens(input_pdb, output_pdb):
    """Remove hydrogen atoms from PDB file"""
    # Read PDB
    mol = Chem.MolFromPDBFile(input_pdb, removeHs=False)

    if mol is None:
        print(f"Error: Cannot read PDB file: {input_pdb}")
        return False

    # Remove hydrogens
    mol_no_h = Chem.RemoveHs(mol)

    # Write PDB
    writer = Chem.PDBWriter(output_pdb)
    writer.write(mol_no_h)
    writer.close()

    print(f"Success! {mol.GetNumAtoms()} atoms -> {mol_no_h.GetNumAtoms()} atoms")
    print(f"Output saved to: {output_pdb}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python remove_h.py input.pdb output.pdb")
        sys.exit(1)

    input_pdb = sys.argv[1]
    output_pdb = sys.argv[2]

    remove_hydrogens(input_pdb, output_pdb)
