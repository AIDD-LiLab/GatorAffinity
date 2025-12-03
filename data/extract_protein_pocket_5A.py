#!/usr/bin/env python3
"""
Simple script to extract protein pocket (5A cutoff) from protein and ligand PDB files

Usage:
    python extract_protein_pocket_5A.py protein.pdb ligand.pdb pocket_5A.pdb
    python extract_protein_pocket_5A.py protein.pdb ligand.pdb pocket_5A.pdb --distance 6.0
"""

import argparse
from pathlib import Path
import pymol
from pymol import cmd

# Initialize PyMOL in quiet mode
pymol.finish_launching(['pymol', '-cq'])


def extract_protein_pocket(protein_path, ligand_path, output_path, distance_threshold=5.0):
    """
    Extract protein pocket using PyMOL based on distance to ligand

    Args:
        protein_path: Path to protein PDB file
        ligand_path: Path to ligand PDB/SDF/MOL2 file
        output_path: Path to save pocket PDB
        distance_threshold: Distance cutoff in Angstroms (default: 5.0)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Protein: {protein_path}")
        print(f"Ligand:  {ligand_path}")
        print(f"Distance: {distance_threshold} A")

        # Clear PyMOL
        cmd.reinitialize()

        # Load protein
        cmd.load(str(protein_path), 'protein')
        cmd.remove('resn HOH+WAT')  # Remove water

        # Load ligand
        cmd.load(str(ligand_path), 'ligand')

        # Select pocket: protein residues within distance threshold of ligand
        cmd.select(
            'pocket_residues',
            f'byres (ligand around {distance_threshold}) and polymer.protein'
        )

        # Check if any atoms were selected
        pocket_atoms = cmd.count_atoms('pocket_residues')

        if pocket_atoms == 0:
            print(f"\nERROR: No protein atoms found within {distance_threshold}A of ligand!")
            cmd.delete('all')
            return False

        print(f"Pocket atoms: {pocket_atoms}")

        # Save pocket
        cmd.save(str(output_path), 'pocket_residues')
        cmd.delete('all')

        # Verify output
        if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            file_size = Path(output_path).stat().st_size / 1024
            print(f"Output: {output_path} ({file_size:.1f} KB)")
            print("SUCCESS!")
            return True
        else:
            print("ERROR: Failed to create output file")
            return False

    except Exception as e:
        print(f"\nERROR: {e}")
        try:
            cmd.delete('all')
        except:
            pass
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Extract protein pocket (5A cutoff)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_protein_pocket_5A.py protein.pdb ligand.pdb pocket_5A.pdb
  python extract_protein_pocket_5A.py protein.pdb ligand.sdf pocket_6A.pdb --distance 6.0
"""
    )

    parser.add_argument('protein', help='Protein PDB file')
    parser.add_argument('ligand', help='Ligand file (PDB/SDF/MOL2)')
    parser.add_argument('output', help='Output pocket PDB file')
    parser.add_argument('--distance', '-d', type=float, default=5.0,
                        help='Distance cutoff in Angstroms (default: 5.0)')

    args = parser.parse_args()

    # Check input files
    protein_path = Path(args.protein)
    ligand_path = Path(args.ligand)
    output_path = Path(args.output)

    if not protein_path.exists():
        print(f"ERROR: Protein file not found: {protein_path}")
        return 1

    if not ligand_path.exists():
        print(f"ERROR: Ligand file not found: {ligand_path}")
        return 1

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXTRACTING PROTEIN POCKET")
    print("=" * 60)

    success = extract_protein_pocket(
        protein_path=protein_path,
        ligand_path=ligand_path,
        output_path=output_path,
        distance_threshold=args.distance
    )

    print("=" * 60)
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
