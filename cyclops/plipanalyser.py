import os
import glob
import urllib.request
import subprocess
import mdtraj as md
import pandas as pd
from Bio import PDB, pairwise2
from Bio.SeqUtils import seq1
import xml.etree.ElementTree as ET


CUSTOM_MODS = {"DYE": "C"} #to accomodate the PTM 


DEFAULT_FASTA = "".join("""
MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECH
KKYGKVWGFYDGQQPVLAITDPDMIKTVLVKECYSVFTNRRPFGPVGFMKSAISIAEDEEWKRLRSLL
SPTFTSGKLKEMVPIIAQYGDVLVRNLRREAETGKPVTLKDVFGAYSMDVITSTSFGVNIDSLNNPQD
PFVENTKKLLRFDFLDPFFLSITVFPFLIPILEVLNICVFPREVTNFLRKSVKRMKESRLEDTQKHRV
DFLQLMIDSQNSKETESHKALSDLELVAQSIIFIFAGYETTSSVLSFIMYELATHPDVQQKLQEEIDA
VLPNKAPPTYDTVLQMEYLDMVVNETLRLFPIAMRLERVCKKDVEINGMFIPKGVVVMIPSYALHRDP
KYWTEPEKFLPERFSKKNKDNIDPYIYTPFGSGPRNCIGMRFALMNMKLALIRVLQNFSFKPCKETQI
PLKLSLGGLLQPEKPVVLKVESRDGTVSGA
""".split())    #CYP 3A4 fasta

def parse_plip_xml(xml_file_path, target_ligand=None, first_site_only=False):
    """
    Parses a PLIP XML report and extracts interactions.
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error loading XML: {e}")
        return []

    interactions_list = []
    sites_processed_count = 0

#may need to accomodate the fact that we have sometimes more than a single pocket - ligand example per pdb, to revisit
    for site in root.findall('bindingsite'):
        

        if first_site_only and sites_processed_count >= 1:
            break

        interaction_group = site.find('interactions')
        if interaction_group is None:
            continue

        valid_interactions_in_site = []

        for int_type_group in interaction_group:
            interaction_type_name = int_type_group.tag 
            
            for interaction in int_type_group:
                
                # only the ligand we want to analyse
                ligand_part_name = interaction.find('restype_lig').text

                if target_ligand and ligand_part_name != target_ligand:
                    continue
                
                # Ignore HEM by default if no target specified, may want to add DYE to this
                if not target_ligand and ligand_part_name == 'HEM':
                    continue

                
                resnr = interaction.find('resnr').text
                reschain = interaction.find('reschain').text
                restype = interaction.find('restype').text
                
                dist = interaction.find('dist')
                if dist is None: dist = interaction.find('dist_h_a')
                dist_val = dist.text if dist is not None else "N/A"

                interaction_data = {
                    'ligand_fragment': ligand_part_name, 
                    'type': interaction_type_name,
                    'residue_name': restype,
                    'residue_number': resnr,
                    'residue_chain': reschain,
                    'residue_full': f"{restype} {reschain}-{resnr}",
                    'distance': dist_val,
                    'id': interaction.get('id')
                }
                
                valid_interactions_in_site.append(interaction_data)


        if valid_interactions_in_site:
            interactions_list.extend(valid_interactions_in_site)
            sites_processed_count += 1

    return interactions_list

def sanitize_pdb_topology(in_file, out_file, verbose=True):
    """
    Fallback: Reads a large PDB (with >99,999 atoms) and rewrites atom serials. Legacy, was used to accomodate mdanalysis. 
    """
    if verbose: print(f"Sanitizing Topology: {in_file} -> {out_file}")
    with open(in_file, 'r') as f_in, open(out_file, 'w') as f_out:
        atom_serial = 1
        for line in f_in:
            if line.startswith(("ATOM", "HETATM")):
                safe_serial = (atom_serial % 99999)
                if safe_serial == 0: safe_serial = 99999
                serial_str = f"{safe_serial:5d}"
                new_line = line[:6] + serial_str + line[11:]
                f_out.write(new_line)
                atom_serial += 1
            elif line.startswith("CONECT"):
                continue
            else:
                f_out.write(line)
    return out_file

def _resname_to_one(resname: str) -> str:
    name = resname.strip().upper()
    try:
        return seq1(name, custom_map=CUSTOM_MODS)
    except TypeError:
        return CUSTOM_MODS.get(name, seq1(name))

def _get_chain_data(chain):
    """Extracts (Sequence, List_of_Residue_Objects) for a chain."""
    residues = []
    for res in chain.get_residues():
        resname = res.get_resname().strip().upper()
        if PDB.Polypeptide.is_aa(res, standard=False) or resname in CUSTOM_MODS:
             residues.append(res)
             
    seq_str = "".join(_resname_to_one(res.get_resname()) for res in residues)
    return seq_str, residues

def renumber_pdb_to_fasta(pdb_file, fasta_seq, out_file="aligned_frame.pdb", verbose=False):
    """
    Aligns PDB sequence to a target FASTA string and renumbers PDB residues 
    to match the FASTA index (1-based).
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_file)
    model = next(structure.get_models())
    
    if 'A' in model:
        chain = model['A']
    else:
        chain = next(model.get_chains())

    pdb_seq, pdb_residues = _get_chain_data(chain)
    target_seq = "".join(fasta_seq.split())


    aln = pairwise2.align.globalxx(pdb_seq, target_seq)[0]
    seqA_pdb, seqB_fasta = aln.seqA, aln.seqB

    if verbose:
        matches = sum(1 for a, b in zip(seqA_pdb, seqB_fasta) if a == b and a != '-')
        align_len = len(seqA_pdb)
        identity = (matches / align_len * 100) if align_len > 0 else 0
        
        print(f"   [Align Check] Identity: {identity:.1f}% ({matches}/{align_len} residues)")
        if identity < 85.0:
            print(f"  Low identity! Renumbering may be incorrect.")

    mapping = []
    i_pdb = 0
    i_fasta = 0 
    
    for char_pdb, char_fasta in zip(seqA_pdb, seqB_fasta):
        if char_pdb != "-" and char_fasta != "-":
            target_res_id = i_fasta + 1
            mapping.append((pdb_residues[i_pdb], target_res_id))
            i_pdb += 1
            i_fasta += 1
        elif char_pdb != "-" and char_fasta == "-":
            i_pdb += 1
        elif char_pdb == "-" and char_fasta != "-":
            i_fasta += 1


    if verbose and len(mapping) > 0:
        indices = [0, len(mapping)//2, len(mapping)-1]
        check_msg = []
        for idx in indices:
            res, target_id = mapping[idx]
            check_msg.append(f"{res.get_resname()}{res.id[1]}->{target_id}")
        print(f"   [Spot Check] {', '.join(check_msg)}")


    current_nums = [r.id[1] for r in chain.get_residues()]
    max_num = max(current_nums) if current_nums else 0
    TEMP_OFFSET = max_num + 50000
    

        
    for idx, (res, _) in enumerate(mapping):
        het, _, icode = res.id
        res.id = (het, TEMP_OFFSET + idx, icode)

    for res, target_num in mapping:
        het, _, icode = res.id
        res.id = (het, int(target_num), icode)

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(out_file)
    return out_file

class MDPlipAnalysis:
    """
    Analyzes protein-ligand interactions across Molecular Dynamics (MD) trajectories using PLIP.

    This class automates the process of:
    1. Downloading and preparing a reference Crystal structure.
    2. Renumbering trajectory frames to match a canonical sequence (FASTA).
    3. Running PLIP (Protein-Ligand Interaction Profiler) on individual frames.
    4. Comparing dynamic interactions against the static reference structure to track stability.

    Parameters
    ----------
    pdb_id : str
        The RCSB PDB ID of the reference structure (e.g., '1TQN').
    ligand_code : str
        The 3-letter residue name of the ligand to analyze (e.g., 'HEM', 'LIG').
    working_dir : str, optional
        Directory where temporary PDBs and XML reports will be stored. Default is current dir.
    verbose : bool, optional
        If True, prints detailed progress logs. Default is True.
    ref_fasta : str, optional
        A specific amino acid sequence to align and renumber the structure against. 
        If None, uses a default internal sequence (CYP3A4).
    renumber_ref : bool, optional
        Whether to renumber the reference PDB file before analysis. Default is False.
    renumber_sim : bool, optional
        Whether to renumber every frame of the simulation to match the reference. 
        Critical for comparing MD frames to Crystal structures. Default is True.
    """
    def __init__(self, pdb_id, ligand_code, working_dir=".", verbose=True, ref_fasta=None, 
             renumber_ref=False, renumber_sim=True): 
        self.pdb_id = pdb_id
        self.ligand_code = ligand_code
        self.working_dir = working_dir
        self.verbose = verbose
        
        self.renumber_ref = renumber_ref
        self.renumber_sim = renumber_sim
        
        # Use provided FASTA or fallback to default
        self.ref_fasta = ref_fasta if ref_fasta else DEFAULT_FASTA
        
        self.ref_interactions = set()
        
    def log(self, message, end="\n"):
        if self.verbose:
            print(message, end=end)

    def _download_pdb_manual(self, pdb_id, out_path):
        """Robust download using Python urllib instead of PLIP CLI."""
        #plip cli was sometimes randomly hanging on NOBUNAGA, probably due to the port restrictions...
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            self.log(f"Downloading {pdb_id} from RCSB...", end=" ")
            urllib.request.urlretrieve(url, out_path)
            self.log("Success.")
            return True
        except Exception as e:
            self.log(f"\nFailed to download PDB: {e}")
            return False

    def step1_prepare_reference(self):
        """
        Manually downloads the PDB.
        Runs PLIP on the structure.
        """
        self.log(f"--- Step 1: Reference ({self.pdb_id}) ---")
        
        raw_ref_pdb = os.path.join(self.working_dir, f"{self.pdb_id.upper()}.pdb")
        renumbered_ref_pdb = os.path.join(self.working_dir, f"{self.pdb_id.upper()}_aligned.pdb")
        
        if not os.path.exists(raw_ref_pdb):
            success = self._download_pdb_manual(self.pdb_id, raw_ref_pdb)
            if not success:
                raise FileNotFoundError(f"Could not retrieve {self.pdb_id} from RCSB.")
        else:
            self.log(f"Using existing reference file: {raw_ref_pdb}")

        # LOGIC: Only renumber reference if explicitly requested
        plip_input_pdb = raw_ref_pdb
        
        if self.renumber_ref:
            self.log("Aligning Reference PDB to Target FASTA...")
            renumber_pdb_to_fasta(raw_ref_pdb, self.ref_fasta, out_file=renumbered_ref_pdb, verbose=self.verbose)
            plip_input_pdb = renumbered_ref_pdb
        else:
            self.log("Using Reference PDB as-is (Original Crystal Numbering).")

        self.log(f"Running PLIP on {plip_input_pdb}...")
        ref_xml = self._run_plip(input_val=plip_input_pdb, mode='file')
        
        if not ref_xml:
            raise FileNotFoundError("PLIP failed to generate an XML report for the reference.")

        ints = parse_plip_xml(ref_xml, target_ligand=self.ligand_code, first_site_only=True)
        self.ref_interactions = self._interactions_to_set(ints)
        self.log(f"Reference Baseline: {len(self.ref_interactions)} interactions found.")
        
        return self.ref_interactions

    def step2_analyze_trajectory(self, trajectory_file, topology_file, stride=1, analyze_fraction=1.0, recenter=True):
        """
        Analyzes trajectory.
        
        Args:
            recenter (bool): If True, unwraps molecules and superposes protein to reference
                             to fix PBC 'jumping' issues before calculating interactions.
        """
        self.log(f"\n--- Step 2: Trajectory Analysis ---")
        self.log(f"Topology: {topology_file}")
        self.log(f"Trajectory: {trajectory_file}")
        self.log(f"Recenter/Unwrap: {recenter}")
        
        # Check frames logic
        try:
            with md.open(trajectory_file) as f:
                total_frames_in_file = len(f)
        except Exception:
            self.log("Warning: Could not determine exact trajectory length. Assuming full read.")
            total_frames_in_file = 0
            
        start_frame_idx = 0
        if total_frames_in_file > 0 and analyze_fraction < 1.0:
            drop_ratio = 1.0 - analyze_fraction
            start_frame_idx = int(total_frames_in_file * drop_ratio)
            self.log(f"Analysis Fraction {analyze_fraction}: Dropping first {start_frame_idx} frames.")
        
        
        try:
            loader = md.iterload(trajectory_file, top=topology_file, stride=stride, chunk=1)
        except Exception as e:
            self.log(f"Error loading topology: {e}")
            if "atom" in str(e).lower():
                self.log("Trying to sanitize PDB (fixing large atom counts)...")
                clean_topo = os.path.join(self.working_dir, "sanitized_topo.pdb")
                sanitize_pdb_topology(topology_file, clean_topo, verbose=self.verbose)
                loader = md.iterload(trajectory_file, top=clean_topo, stride=stride, chunk=1)
            else:
                raise e

        # Selection
        ref_frame = md.load_frame(trajectory_file, top=topology_file, index=0)
        top = ref_frame.topology
        query = f"not resname HEM"
        self.log(f"Selection Query: '{query}'")
        try:
            atom_indices = top.select(query)
        except Exception:
            self.log(f"Selection failed, fallback to 'not water'")
            atom_indices = top.select("not water")
        self.log(f"Selected {len(atom_indices)} atoms.")
        
        results = []
        global_frame_counter = 0 
        
        for chunk in loader:
            for frame in chunk:
                current_frame_id = global_frame_counter * stride
                global_frame_counter += 1
                
                if current_frame_id < start_frame_idx:
                    continue

                if self.verbose:
                    print(f"Processing Frame {current_frame_id}...", end="\r")
                
                # RECENTERING LOGIC
                if recenter:
                    frame.image_molecules(inplace=True)
                    prot_ca = top.select("protein and name CA")
                    if len(prot_ca) > 0:
                        frame.superpose(ref_frame, atom_indices=prot_ca)
                    else:
                        prot_all = top.select("protein")
                        frame.superpose(ref_frame, atom_indices=prot_all)

                sub_traj = frame.atom_slice(atom_indices)
                temp_raw = os.path.join(self.working_dir, f"temp_frame_{current_frame_id}.pdb")
                temp_aligned = os.path.join(self.working_dir, f"aligned_frame_{current_frame_id}.pdb")
                
                sub_traj.save_pdb(temp_raw)
                plip_input = temp_raw

                if self.renumber_sim:
                    renumber_pdb_to_fasta(temp_raw, self.ref_fasta, out_file=temp_aligned, verbose=self.verbose)
                    plip_input = temp_aligned
                
                frame_xml = self._run_plip(input_val=plip_input, mode='file', break_composite=True)
                
                if frame_xml:
                    frame_ints = parse_plip_xml(frame_xml, target_ligand=self.ligand_code)
                    frame_set = self._interactions_to_set(frame_ints)
                    
                    tp = frame_set.intersection(self.ref_interactions)
                    fp = frame_set.difference(self.ref_interactions)
                    fn = self.ref_interactions.difference(frame_set)
                    
                    results.append({
                        'Frame': current_frame_id,
                        'Conserved': len(tp),
                        'New': len(fp),
                        'Lost': len(fn),
                        'Conserved_Details': "; ".join(tp),
                        'New_Details': "; ".join(fp)
                    })
                
                for f in [temp_raw, temp_aligned]:
                    if f and os.path.exists(f): os.remove(f)

        if self.verbose: print() 
        return pd.DataFrame(results)

    def cleanup_temporary_files(self):
        self.log(f"\nCleaning up temporary files in {self.working_dir}...")
        patterns = [
            "temp_frame_*.pdb",
            "aligned_frame_*.pdb",
            "sanitized_*.pdb",
            "plip*.pdb",
        ]
        
        count = 0
        for pat in patterns:
            full_pat = os.path.join(self.working_dir, pat)
            for f in glob.glob(full_pat):
                try:
                    os.remove(f)
                    count += 1
                except OSError:
                    pass
        self.log(f"Removed {count} temporary artifacts.")

    def _run_plip(self, input_val, mode='file', break_composite=False):
        cmd = ["plip"]
        if mode == 'fetch':
            cmd.extend(["-i", input_val]) 
        else:
            cmd.extend(["-f", input_val]) 
            
        cmd.extend(["-x", "-o", self.working_dir])
        if break_composite: cmd.append("--breakcomposite")
            
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if mode == 'fetch':
            candidates = [
                os.path.join(self.working_dir, f"{input_val.upper()}_report.xml"),
                os.path.join(self.working_dir, f"{input_val.lower()}_report.xml"),
            ]
        else:
            base = os.path.basename(input_val).replace(".pdb", "")
            candidates = [
                os.path.join(self.working_dir, f"{base}_report.xml"),
                os.path.join(self.working_dir, "report.xml")
            ]

        for c in candidates:
            if os.path.exists(c):
                if "report.xml" in c and c != candidates[0] and mode == 'file':
                     final_name = candidates[0]
                     os.rename(c, final_name)
                     return final_name
                return c
        return None

    def _interactions_to_set(self, interaction_list):
        s = set()
        for i in interaction_list:
            sig = f"{i['residue_name']} {i['residue_chain']}-{i['residue_number']} : {i['type']}"
            s.add(sig)
        return s