import mdtraj as md
import os
import numpy as np

class RFAAFixer:
    """
    A tool for repairing structural gaps in RFAA-generated PDB files.
    
    This class detects missing residues (specifically near Iron/Heme sites),
    splits hybrid ligand residues into their components (e.g., CYS + HEM),
    and renumbers the sequence to close the structural gap.

    Parameters
    ----------
    pdb_path : str
        The file path to the input PDB file to be fixed.
    """
    def __init__(self, pdb_path):
        self.pdb_path = pdb_path
        
        # Mapping RFAA LG1 atom names to Standard PDB CYS atom names
        self.cys_map = {
            'N1': 'N',
            'C1': 'CA',
            'C2': 'C',
            'O1': 'O',
            'C3': 'CB',
            'S1': 'SG'
        }

    def _parse_pdb_line(self, line):
        """Helper to parse strict PDB columns safely."""
        # Skip lines that are too short to contain atom info
        if len(line) < 27:
            return None

        return {
            'line': line, # Store original line for reference if needed
            'record': line[0:6].strip(),
            'serial': int(line[6:11].strip()) if line[6:11].strip() else 0,
            'name': line[12:16].strip(),
            'altLoc': line[16],
            'resName': line[17:20].strip(),
            'chainID': line[21],
            'resSeq': int(line[22:26].strip()) if line[22:26].strip() else 0,
            'remainder': line[26:]
        }

    def _format_atom_line(self, serial, name, res_name, chain, res_seq, remainder, record="ATOM", altLoc=" "):
        """Helper to format strict PDB columns."""
        # Ensure atom name is padded correctly (e.g. " CA " vs "N   ")
        clean_name = name.strip()
        if len(clean_name) >= 4:
            atom_field = clean_name[:4]
        else:
            atom_field = f" {clean_name}".ljust(4)

        # 2. Residue Name (Cols 18-20, Width 3)
        res_field = res_name.strip().rjust(3)

        # 3. AltLoc (Col 17, Width 1)
        # Ensure it's exactly 1 char, default to space
        alt_field = altLoc if altLoc and len(altLoc) == 1 else " "

        # 4. Construct Line
        # Record(6) + Serial(5) + Space(1) + AtomName(4) + AltLoc(1) + ResName(3) + ...
        return (f"{record.ljust(6)}{str(serial).rjust(5)} {atom_field}{alt_field}{res_field} "
                f"{chain}{str(res_seq).rjust(4)}{remainder}")

    def find_gap_residue(self, cutoff=1.5):
        """
        Finds the residue number immediately BEFORE the structural gap in Chain A,
        specifically searching only near the Iron (Fe) atom.
        """
        t = md.load(self.pdb_path)
        frame = t[0]

        # 1. Select CA atoms
        all_ca_indices = frame.topology.select("protein and name CA")
        if len(all_ca_indices) == 0:
            print("Error: No CA atoms found.")
            return None
            
        # 2. Map atom indices to Residue objects
        ca_atom_to_res = {idx: frame.topology.atom(idx).residue for idx in all_ca_indices}
        res_to_ca_atom_idx = {res.index: atom_idx for atom_idx, res in ca_atom_to_res.items()}

        # 3. Find Iron (or fallback to LG1)
        fe_indices = frame.topology.select("element Fe")
        if len(fe_indices) == 0:
            print("Warning: No 'Fe' atoms found. Trying to find center of LG1...")
            fe_indices = frame.topology.select("resname LG1")
            if len(fe_indices) == 0:
                print("Error: Neither Fe nor LG1 found.")
                return None

        # 4. Find protein residues near the Iron (cutoff in nm)
        nearby_ca_indices = md.compute_neighbors(frame, cutoff, fe_indices, haystack_indices=all_ca_indices)[0]
        
        # Get unique list of residues near the iron, sorted by index
        nearby_residues = sorted(list(set(ca_atom_to_res[idx] for idx in nearby_ca_indices)), key=lambda r: r.index)

        if not nearby_residues:
            print(f"No protein residues found within {cutoff} nm of Iron.")
            return None

        # 5. Find pairs of SEQUENTIAL residues in this neighborhood
        atom_pairs = []
        pair_res_objects = [] # To track which residue corresponds to the gap

        for res1 in nearby_residues:
            # We look for the geometrically next residue (index + 1)
            if (res1.index + 1) in res_to_ca_atom_idx:
                res2 = frame.topology.residue(res1.index + 1)
                
                # If both are in the neighborhood (or just res1 is fine, but we need coords for both)
                # We strictly check distance between res1 and res2
                a1_idx = res_to_ca_atom_idx[res1.index]
                a2_idx = res_to_ca_atom_idx[res2.index]
                
                atom_pairs.append([a1_idx, a2_idx])
                pair_res_objects.append(res1)

        if not atom_pairs:
            print("No sequential residue pairs found near Iron.")
            return None

        # 6. Compute distances for these specific pairs
        dists = md.compute_distances(frame, atom_pairs, periodic=False)[0]
        
        # Find the index of the pair with the largest distance
        max_idx = np.argmax(dists)
        max_gap = dists[max_idx]
        
        # Validate gap size (standard peptide bond CA-CA is ~0.38 nm)
        if max_gap < 0.45: # 4.5 Angstroms
            print(f"Max gap near Iron is only {max_gap*10:.2f} Å. Likely no missing residue.")
            return None
            
        # Get the residue BEFORE the gap
        gap_residue = pair_res_objects[max_idx]
        
        print(f"Gap found between residue {gap_residue.resSeq} and {gap_residue.resSeq+1} (Dist: {max_gap*10:.2f} Å)")
        return gap_residue.resSeq

    def split_and_repair(self, output_path, hybrid_resname="LG1"):
        """
        1. Parses file once.
        2. Splits Hybrid residue into CYS (Chain A) and HEM (Chain B).
        3. Moves original Ligand to Chain C.
        4. Renumbers everything sequentially.
        """
        gap_res_seq = self.find_gap_residue()
        
        if gap_res_seq is None:
            print("Gap detection failed or no gap present. Aborting repair.")
            return

        print(f"Gap detected after residue {gap_res_seq}. Splitting and Repairing...")

        with open(self.pdb_path, 'r') as f:
            lines = f.readlines()

        # --- Containers for Sorted Atoms ---
        protein_pre_gap = []
        protein_post_gap = []
        new_cys_atoms = []
        heme_atoms = []
        ligand_atoms = []

        found_hybrid = False

        # --- Single Pass Classification ---
        for line in lines:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            
            p = self._parse_pdb_line(line)
            if not p: continue 

            # 1. Handle the Hybrid Residue (Chain A, LG1)
            if p['resName'] == hybrid_resname and p['chainID'] == 'A':
                found_hybrid = True
                
                # Check if this atom belongs to Cysteine
                if p['name'] in self.cys_map:
                    # Convert mapping (e.g. N1 -> N)
                    p['name'] = self.cys_map[p['name']]
                    new_cys_atoms.append(p)
                else:
                    # This belongs to Heme
                    heme_atoms.append(p)
            
            # 2. Handle Standard Protein Chain A
            elif p['chainID'] == 'A':
                if p['resSeq'] <= gap_res_seq:
                    protein_pre_gap.append(p)
                else:
                    # This is protein AFTER the gap
                    protein_post_gap.append(p)
            
            # 3. Handle Other Ligands (Chain B in input)
            elif p['chainID'] != 'A':
                ligand_atoms.append(p)

        if not found_hybrid:
            print(f"Error: Could not find {hybrid_resname} in Chain A.")
            return

        # --- Writing & Renumbering ---
        
        final_lines = []
        serial_counter = 1
        
        # 1. Write Protein (Pre-Gap)
        for p in protein_pre_gap:
            final_lines.append(self._format_atom_line(serial_counter, p['name'], p['resName'], 'A', p['resSeq'], p['remainder'], altLoc=p['altLoc']))
            serial_counter += 1

        # 2. Write New CYS
        cys_res_seq = gap_res_seq + 1
        for p in new_cys_atoms:
            final_lines.append(self._format_atom_line(serial_counter, p['name'], "CYS", 'A', cys_res_seq, p['remainder'], record="ATOM"))
            serial_counter += 1

        # 3. Write Protein (Post-Gap) - Increment Residue ID by 1
        for p in protein_post_gap:
            new_seq = p['resSeq'] + 1
            final_lines.append(self._format_atom_line(serial_counter, p['name'], p['resName'], 'A', new_seq, p['remainder'], altLoc=p['altLoc']))
            serial_counter += 1

        # Terminate Chain A
        last_prot = protein_post_gap[-1] if protein_post_gap else new_cys_atoms[-1]
        last_seq = last_prot['resSeq'] + 1 if protein_post_gap else cys_res_seq
        
        # TER format
        final_lines.append(f"TER   {str(serial_counter).rjust(5)}      {last_prot['resName'].rjust(3)} A{str(last_seq).rjust(4)}                                                      \n")
        serial_counter += 1

        # 4. Write Heme (Chain B)
        heme_res_seq = 1 
        for p in heme_atoms:
            final_lines.append(self._format_atom_line(serial_counter, p['name'], "HEM", 'B', heme_res_seq, p['remainder'], record="HETATM", altLoc=p['altLoc']))
            serial_counter += 1
        
        final_lines.append(f"TER   {str(serial_counter).rjust(5)}      HEM B{str(heme_res_seq).rjust(4)}                                                      \n")
        serial_counter += 1

        # 5. Write Ligand (Chain C)
        lig_res_seq = 1
        for p in ligand_atoms:
            final_lines.append(self._format_atom_line(serial_counter, p['name'], "LIG", 'C', lig_res_seq, p['remainder'], record="HETATM", altLoc=p['altLoc']))
            serial_counter += 1

        final_lines.append("END\n")

        with open(output_path, 'w') as f:
            f.writelines(final_lines)
            
        print(f"Successfully wrote: {output_path}")