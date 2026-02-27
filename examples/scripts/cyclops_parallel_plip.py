import os
import glob
import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from cyclops.plipanalyser import MDPlipAnalysis

# --- Configuration for Analysis ---
# You can adjust these default values
STRIDE = 1
ANALYZE_FRACTION = 0.9

def analyze_system(args):
    """
    Worker function for parallel processing.
    args: tuple (target_dir, pdb_id, ligand_id)
    """
    target_dir, pdb_id, ligand_id = args
    
    traj_file = os.path.join(target_dir, "trajectory_complex.xtc")
    topo_file = os.path.join(target_dir, "topology_complex.pdb")
    output_csv = os.path.join(target_dir, "plip_md_results.csv")

    # Check if files exist
    if not os.path.exists(traj_file) or not os.path.exists(topo_file):
        return f"[{pdb_id}] SKIPPED: Missing trajectory or topology in {os.path.basename(target_dir)}"

    # Check if already done (optional, prevents re-running)
    #if os.path.exists(output_csv):
    #    return f"[{pdb_id}] SKIPPED: Results already exist in {os.path.basename(target_dir)}"

    print(f"[{pdb_id}] Starting analysis in {target_dir}...")

    try:
        # Initialize the pipeline
        pipeline = MDPlipAnalysis(
            pdb_id, 
            ligand_id, 
            working_dir=target_dir, # CRITICAL: This keeps files in the subdir
            verbose=False, 
            renumber_ref=False, 
            renumber_sim=True
        )

        # Step 1: Reference
        pipeline.step1_prepare_reference()

        # Step 2: Trajectory
        df = pipeline.step2_analyze_trajectory(
            traj_file,
            topo_file,
            stride=STRIDE,
            analyze_fraction=ANALYZE_FRACTION
        )

        if not df.empty:
            df.to_csv(output_csv, index=False)
            msg = f"[{pdb_id}] SUCCESS: Saved {len(df)} frames to {output_csv}"
        else:
            msg = f"[{pdb_id}] WARNING: Analysis ran but returned empty DataFrame."

        pipeline.cleanup_temporary_files()
        return msg

    except Exception as e:
        # Catch errors so one failure doesn't crash the whole batch
        return f"[{pdb_id}] ERROR in {os.path.basename(target_dir)}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Batch PLIP Analysis for MD Trajectories")
    parser.add_argument("--csv", required=True, help="Path to ligand_descriptors.csv")
    parser.add_argument("--dir", required=True, help="Main directory containing subfolders *-{PDB}")
    parser.add_argument("--cores", type=int, default=4, help="Number of parallel processes")
    
    args = parser.parse_args()

    # 1. Load CSV into a Lookup Map
    try:
        df_csv = pd.read_csv(args.csv)
        required_cols = ['PDB ID', 'LigID']
        if not all(col in df_csv.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        # Create a dictionary: { '3UA5': '06X', ... }
        # We enforce Uppercase for PDB IDs to ensure matching works
        pdb_ligand_map = {}
        for _, row in df_csv.iterrows():
            p_id = str(row['PDB ID']).strip().upper()
            l_id = str(row['LigID']).strip()
            pdb_ligand_map[p_id] = l_id
            
        print(f"Loaded {len(pdb_ligand_map)} PDB-Ligand pairs from CSV.")
        
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    # 2. Scan Directory for Simulation Folders
    if not os.path.exists(args.dir):
        print(f"Error: Directory '{args.dir}' does not exist.")
        return

    all_items = glob.glob(os.path.join(args.dir, "*"))
    subdirs = [d for d in all_items if os.path.isdir(d)]
    
    tasks = []
    
    print(f"Scanning {len(subdirs)} subdirectories in '{args.dir}'...")

    for d in subdirs:
        dirname = os.path.basename(d)
        
        # Parse PDB ID from folder name pattern "*-{PDB}"
        # We assume the PDB ID is the last part after the hyphen
        if "-" in dirname:
            candidate_pdb = dirname.split("-")[-1].upper()
            
            # Check if this PDB exists in our CSV map
            if candidate_pdb in pdb_ligand_map:
                ligand_id = pdb_ligand_map[candidate_pdb]
                tasks.append((d, candidate_pdb, ligand_id))
            else:
                # Optional: Warn if folder looks like a target but isn't in CSV
                # print(f"Skipping {dirname}: PDB '{candidate_pdb}' not found in CSV.")
                pass
        else:
            # Fallback for exact matches if folder is named just "3UA5"
            candidate_pdb = dirname.upper()
            if candidate_pdb in pdb_ligand_map:
                ligand_id = pdb_ligand_map[candidate_pdb]
                tasks.append((d, candidate_pdb, ligand_id))

    print(f"Found {len(tasks)} matching simulations to process.")
    print(f"Running on {args.cores} cores.")
    print("-" * 50)

    # 3. Run Parallel
    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        # Submit all tasks
        # Task tuple is (target_dir, pdb_id, ligand_id)
        future_to_pdb = {executor.submit(analyze_system, task): task[1] for task in tasks}

        for future in as_completed(future_to_pdb):
            pdb = future_to_pdb[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f"[{pdb}] CRITICAL EXCEPTION: {exc}")

if __name__ == "__main__":
    main()
