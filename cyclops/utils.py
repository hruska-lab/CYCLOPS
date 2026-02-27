import mdtraj
import openmm
import openmm.app
import openmm.unit
from openff.toolkit import Molecule, unit, Topology
from openff.interchange import Interchange
from openff.interchange.components._packmol import RHOMBIC_DODECAHEDRON, pack_box, UNIT_CUBE
from openmm.app import PDBFile, Modeller, ForceField, Simulation, StateDataReporter
from openmm import LangevinIntegrator
from openmm.unit import picoseconds, kelvin, femtoseconds, nanometers
import copy
from pathlib import Path
import requests
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import mdtraj as md
import openmm as mm
import openmm.app as app
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.interchange.drivers import (
    get_amber_energies,
    get_gromacs_energies,
    get_openmm_energies,
)
from openff.interchange.drivers.all import get_summary_data
from openbabel import pybel
from pdbfixer import PDBFixer
import os  
import numpy as np
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Atom import Atom
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Suppress warnings about discontinuous chains, etc.
warnings.simplefilter('ignore', PDBConstructionWarning)

__all__ = [

    "protonate",
    "create_ptm_protein_pdb"
    "fix_protein",
    "remove_crystal_water"
    "create_ptm_protein_pdb"
    "_fix_fe_line_formatting"
]





def extract_non_standard_residues(pdb_file, simulation_name, verbose = False):
    
    complex = md.load(pdb_file)
    
    standard_residues = [
        'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 
        'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 
        'TRP', 'TYR', 'HOH', 'HEM'
    ]
    
    ligand = None
    
    for res in complex.top.residues:
        if res.name not in standard_residues:

            ligand = complex.atom_slice(complex.top.select(f"resname {res.name}"))
            if verbose:
                print(f"Extracted residue {res.name} and saved as ligand.")
            # output_pdb = f'{res.name}_from_{pdb_file.split(".")[0]}.pdb'
            output_pdb = f'{simulation_name}/unprotonated_ligand.pdb'
            ligand.save(output_pdb)
            break  
    
    return ligand

def protonate(infile: str, outname: str, formats: tuple[str, str] = None, polarOnly=False, pH=7.4, verbose = False) -> None:
    """
    Protonates a molecule and saves the protonated structure to a new file of specified format. Uses openbabel for protonation.

    Parameters
    ----------
    infile (str): The path to the input PDB file.
    outname (str): The name of the output file.
    formats (tuple[str, str]): A tuple containing the input and output file formats (e.g. ("pdb", "pdb")). 
    polarOnly (bool, optional): If True, only polar hydrogens are added. Defaults to True.
    pH (float, optional): The pH value to use for protonation. Defaults to 7.4.

    Returns:
        None
    """
    if formats == None:
        formats = (infile.split('.')[-1], outname.split('.')[-1])

    # Load the input PDB file
    mol = next(pybel.readfile(formats[0], infile))

    # Protonate the molecule at pH 7.4
    mol.OBMol.AddHydrogens(polarOnly, True, pH)

    # Write the protonated structure to a new PDB file

    output = pybel.Outputfile(formats[1], outname , overwrite=True)
    output.write(mol)
    output.close()

    if verbose:
        print(f"Protonation completed. Output saved to {outname}")

def extract_protein(pdb_file, simulation_name, verbose = False):
    
    complex = md.load(pdb_file)

    
    standard_residues = [
        'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 
        'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 
        'TRP', 'TYR'
    ]

    protein_atoms = complex.top.select(' or '.join([f'resname {res}' for res in standard_residues]))

    if len(protein_atoms) == 0:
        if verbose:
            print("No protein atoms found in the provided PDB file.")
        return None

    protein = complex.atom_slice(protein_atoms)

    output_pdb = f'{simulation_name}/plain_protein.pdb'
    protein.save(output_pdb)

    if verbose:
        print(f"Extracted protein and saved as '{output_pdb}'.")

    return protein

def extract_heme_fe(pdb_file, simulation_name, verbose = False):
    
    complex = md.load(pdb_file)

    heme_atoms = complex.top.select("resname HEM")
    
    heme = complex.atom_slice(heme_atoms)

    fe_atoms = heme.top.select("element Fe")

    fe = heme.atom_slice(fe_atoms)

    heme_without_fe_atoms = [atom.index for atom in heme.top.atoms if atom.index not in fe_atoms]
    heme_without_fe = heme.atom_slice(heme_without_fe_atoms)
    
    output_heme_pdb = f'{simulation_name}/heme_without_fe_unprotonated.pdb'
    heme_without_fe.save(output_heme_pdb)

    output_fe_pdb = f"{simulation_name}/Fe.pdb"
    fe.save(output_fe_pdb)
    
    if verbose:
        print(f"Extracted heme group without fe and saved as '{output_heme_pdb}' and extracted Fe and saved as '{output_fe_pdb}'.")
    
    return heme_without_fe, fe

def extract_heme_including_fe(pdb_file, simulation_name, verbose = False):
    
    complex = md.load(simulation_name + '/' + pdb_file)

    heme_atoms = complex.top.select("resname HEM")
    
    heme = complex.atom_slice(heme_atoms)


    heme = heme.atom_slice(heme_without_fe_atoms)
    
    output_heme_pdb = f'{simulation_name}/heme_with_fe_unprotonated.pdb'
    heme.save(output_heme_pdb)
    
    if verbose:
        print(f"Extracted heme group without fe and saved as '{output_heme_pdb}''.")
    
    return heme
    
    #depict keyword left for compatibility
def prepare_heme(sdf_file, resname_heme, smiles_heme, simulation_name,  depict=True, output_sdf='prepared_heme.sdf'):
    
    supplier = Chem.SDMolSupplier(str(sdf_file))
    
    rdkit_mol = supplier[0]
    if rdkit_mol is None:
        raise ValueError(f"No valid molecules found in SDF file: {sdf_file}")

    heme = rdkit_mol 
    
    heme = Chem.RemoveHs(heme)
    
    reference_mol_heme = Chem.MolFromSmiles(smiles_heme)
    
    prepared_heme = AllChem.AssignBondOrdersFromTemplate(reference_mol_heme, heme)
    prepared_heme.AddConformer(heme.GetConformer(0))
    
    prepared_heme = Chem.rdmolops.AddHs(prepared_heme, addCoords=True)
    
    with Chem.SDWriter(f'{simulation_name}/{output_sdf}') as writer:
        writer.write(prepared_heme)

    return prepared_heme


def prepare_ligand(unprot_file: str, neutral_smiles: str, output_sdf: str):
    """
    Uses a neutral SMILES template to assign correct bond orders
    to an input molecule file (e.g., from a PDB).
    """
    # --- 1. Load the PDB file ---
    file_extension = os.path.splitext(unprot_file)[1].lower()
    if file_extension == '.pdb':
        mol = Chem.MolFromPDBFile(unprot_file, sanitize=True)
    elif file_extension == '.sdf':
        supplier = Chem.SDMolSupplier(unprot_file, sanitize=True)
        mol = supplier[0] if supplier else None
    else:
        raise ValueError(f"Unsupported file format: {file_extension} for {unprot_file}")

    if mol is None:
        raise ValueError(f"Could not read molecule from {unprot_file}")

    # --- 2. Create the heavy-atom template ---
    mol_noH = Chem.RemoveHs(mol, sanitize=True)
    reference_mol = Chem.MolFromSmiles(neutral_smiles)
    if reference_mol is None:
        raise ValueError(f"RDKit could not parse the reference SMILES: {neutral_smiles}")

    # --- 3. Create the prepared molecule (heavy atoms only) ---
    # This creates a new molecule with M heavy atoms and correct bond orders
    prepared_mol_noH = AllChem.AssignBondOrdersFromTemplate(reference_mol, mol_noH)
    if prepared_mol_noH is None:
        raise ValueError("Failed to assign bond orders from template.")

    # --- 4. Copy the heavy-atom 3D coordinates ---
    # We must create a new conformer for our new molecule
    # and copy the heavy atom positions from the *original* molecule.
    
    # Get the heavy atom indices from the *original* mol
    heavy_atom_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    
    # Check that atom counts match (they should)
    if len(heavy_atom_indices) != prepared_mol_noH.GetNumAtoms():
        raise ValueError("Heavy atom count mismatch after RemoveHs. Check your input PDB and SMILES.")
        
    original_conformer = mol.GetConformer(0)
    new_conformer = Chem.Conformer(prepared_mol_noH.GetNumAtoms())
    
    # Map old heavy atom indices to new atom indices (0...M-1)
    atom_map = {}
    for i, old_idx in enumerate(heavy_atom_indices):
         atom_map[i] = old_idx

    for new_idx in range(prepared_mol_noH.GetNumAtoms()):
        old_idx = atom_map[new_idx]
        new_conformer.SetAtomPosition(new_idx, original_conformer.GetAtomPosition(old_idx))
        
    prepared_mol_noH.AddConformer(new_conformer, assignId=True)

    # --- 5. Add hydrogens with 3D coordinates ---
    # addCoords=True will embed the hydrogens based on the 
    # heavy-atom conformer we just added.
    prepared_mol_withH = Chem.rdmolops.AddHs(prepared_mol_noH, addCoords=True)
    
    # Use RDKit's CoordGen to clean up the H positions
    try:
        AllChem.ConstrainedEmbed(prepared_mol_withH, prepared_mol_noH)
    except Exception as e:
        print(f"Warning: ConstrainedEmbed failed, falling back. Error: {e}")
        # Fallback if constrained embed fails
        AllChem.EmbedMolecule(prepared_mol_withH)

    # --- 6. Write the final molecule ---
    with Chem.SDWriter(output_sdf) as writer:
        writer.write(prepared_mol_withH)
    
    print(f"Fixed bond orders and saved to {output_sdf}")
    return prepared_mol_withH

def pdb_to_sdf(pdb_file: str, sdf_file: str):
    """Converts a PDB file to an SDF file using Open Babel."""
    mol = next(pybel.readfile("pdb", pdb_file))
    mol.write("sdf", sdf_file, overwrite=True)

def fix_protein(
    input_pdb_path: str,
    output_pdb_path: str,
    ph: float = 7.4
):
    """
    Fixes a protein PDB file using PDBFixer.
    
    - Finds and replaces non-standard residues
    - Finds and adds missing atoms
    - Adds missing hydrogens at a specified pH
    """
    fixer = PDBFixer(filename=input_pdb_path)

    modeller = Modeller(fixer.topology, fixer.positions)
    atoms_to_delete = [atom for atom in modeller.topology.atoms() if atom.element.symbol == 'H']
    
    # Delete the specific atoms found above
    modeller.delete(atoms_to_delete)
    
    # Update the fixer object with the clean topology
    fixer.topology = modeller.topology
    fixer.positions = modeller.positions
    
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.addMissingHydrogens(ph)
    
    with open(output_pdb_path, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

def remove_crystal_water(
    protein_topology: "openff.toolkit.Topology",
) -> "openff.toolkit.Topology":
    """Removes 'O' (water) molecules from an OpenFF Topology."""
    protein_only_topology = Topology()
    for molecule in protein_topology.molecules:
        if molecule.to_smiles() != "O":
            protein_only_topology.add_molecule(molecule)
    return protein_only_topology

def _fix_fe_line_formatting(
    filepath: str, 
    atom_name: str = "Fe1x", 
    verbose: bool = False
):
    """
    Post-processes a PDB file to fix a common Biopython formatting bug
    where the element "FE" is written in columns 76-77 instead of 77-78.
    
    This function reads the file, finds the specific HETATM line,
    and inserts a space at column 76 (index 75) to shift "FE"
    to the correct position, truncating the line to 80 chars.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        found_and_fixed = False
        new_lines = []
        
        for line in lines:
            # Check if it's the HETATM line for our specific atom
            # and if the bug is present (FE starting at index 75)
            if (
                line.startswith("HETATM")
                and atom_name in line
                and len(line) >= 77
                and line[75:77] == "FE"
            ):
                if verbose:
                    print(f"  -> Found formatting bug in line: {line.strip()}")
                
                # Reconstruct the line by inserting a space at index 75
                # This shifts "FE" and everything after it right by one
                fixed_line_insert = line[:75] + " " + line[75:]
                
                # Truncate back to 80 characters + newline
                # This drops the last character (which is almost certainly a space)
                fixed_line = fixed_line_insert[:80] + "\n"
                
                new_lines.append(fixed_line)
                found_and_fixed = True
                if verbose:
                    print(f"  -> Fixed line to: {fixed_line.strip()}")
            else:
                # This line is fine, add it as-is
                new_lines.append(line)

        # After checking all lines, write the new content back to the file
        if found_and_fixed:
            with open(filepath, 'w') as f:
                f.writelines(new_lines)
            if verbose:
                print(f"Post-processing: Fixed element formatting in {filepath}")
        elif verbose:
            print(f"Post-processing: No formatting fix was needed for {filepath}.")

    except FileNotFoundError:
        print(f"Error in post-processing: File not found at {filepath}")
    except Exception as e:
        print(f"Error during post-processing: {e}")


def create_ptm_protein_pdb(
    protein_pdb_path: str,
    fe_pdb_path: str,
    output_pdb_path: str,
    new_residue_name: str = "CYF",
    verbose: bool = False
):
    """
    Creates a single PDB file with an embedded PTM (CYS-Fe) residue.
    This function finds the Cysteine (CYS) residue closest to the Iron (Fe)
    atom, removes its 'HG' atom, converts the entire CYS residue to
    HETATM records, renames it, and adds the Fe atom to it.

    Parameters
    ----------
    protein_pdb_path : str
        Path to the PDB file of the full protein.
    fe_pdb_path : str
        Path to the PDB file containing *only* the Fe atom.
    output_pdb_path : str
        File path to save the new, modified protein PDB.
    new_residue_name : str
        The new 3-letter name for the PTM residue (e.g., "CYF").
    verbose : bool
        If True, prints status messages.
    """

    # --- 1. Load Structures ---
    parser = PDBParser(PERMISSIVE=1)
    if verbose:
        print(f"Loading protein from: {protein_pdb_path}")
    protein_struct = parser.get_structure("protein", protein_pdb_path)

    if verbose:
        print(f"Loading Fe atom from: {fe_pdb_path}")
    fe_struct = parser.get_structure("fe", fe_pdb_path)

    # Get the Fe atom object (assume it's the first/only one)
    try:
        fe_atom = next(fe_struct.get_atoms())
        if verbose:
            print(f"Found Fe atom at coords: {fe_atom.get_coord()}")
    except StopIteration:
        print(f"Error: No atoms found in {fe_pdb_path}")
        return

    # --- 2. Find Closest Cysteine ---
    min_dist = float('inf')
    target_residue = None

    if verbose:
        print("Searching for closest Cysteine (CYS) 'SG' atom to the Fe...")

    for residue in protein_struct.get_residues():
        if residue.get_resname() == "CYS":
            try:
                sg_atom = residue['SG']
                distance = np.linalg.norm(sg_atom.get_coord() - fe_atom.get_coord())

                if distance < min_dist:
                    min_dist = distance
                    target_residue = residue
            except KeyError:
                continue

    if target_residue is None:
        print("Error: Could not find any CYS residues in the protein PDB.")
        return

    if verbose:
        print(f"Found target CYS: {target_residue.get_full_id()} at distance {min_dist:.2f} Å")

    # --- 3. Modify the Cysteine Residue ---

    # a. Remove the thiol hydrogen ('HG') if it exists
    if 'HG' in target_residue:
        if verbose:
            print("Removing 'HG' atom from CYS.")
        target_residue.detach_child('HG')

    # b. Change residue to HETATM and rename it
    original_id = target_residue.id
    new_id = (f"H_{new_residue_name}", original_id[1], original_id[2])

    target_residue.id = new_id
    target_residue.resname = new_residue_name
    if verbose:
        print(f"Converted CYS {original_id[1]} to HETATM residue '{new_residue_name}'")

    # --- 4. Add the Fe Atom to the New Residue ---
    
    # Use the same 'Fe1x' name from your example
    new_fe_atom_name = "Fe1x" 
    new_fe_atom = Atom(
        name=new_fe_atom_name,
        coord=fe_atom.get_coord(),
        bfactor=fe_atom.get_bfactor(),
        occupancy=fe_atom.get_occupancy(),
        altloc=fe_atom.get_altloc(),
        fullname=f" {new_fe_atom_name:<3}", # Standard 4-char formatted name
        serial_number=-1,
        element="FE"  # Specify the element
    )

    target_residue.add(new_fe_atom)
    if verbose:
        print(f"Added Fe atom (named '{new_fe_atom_name}') to the {new_residue_name} residue.")


    # --- 5. Save the New PDB File ---
    io = PDBIO()
    io.set_structure(protein_struct)
    io.save(output_pdb_path)

    if verbose:
        print(f"\nSuccessfully saved new PDB to: {output_pdb_path}")

    # --- 6. Run Post-Processing Fix ---
    if verbose:
        print("Running post-processing to fix Fe element formatting...")
    
    # This is the new step that fixes the file you just saved
    _fix_fe_line_formatting(
        filepath=output_pdb_path,
        atom_name=new_fe_atom_name,
        verbose=verbose
    )
    
    if verbose:
        print("--- PTM Creation Complete ---")
        
        