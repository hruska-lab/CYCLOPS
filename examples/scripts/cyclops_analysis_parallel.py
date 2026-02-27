import pandas as pd
import sys
import os
import warnings
import concurrent.futures
import time
import argparse
from cyclops.analysis import HemeAnalysis, aggregate_ligand_statistics

warnings.filterwarnings("ignore")

def process_single_simulation(row_data):
    """
    Worker function to process a single row of data.
    Must be at the module level (not inside main) for multiprocessing.
    """
    index, file_loc, pdb_code, lig_code = row_data

    sim_dir = os.path.join("outputs", file_loc[:9])

    docked_pdb = os.path.join("pdb_inputs", file_loc)

    ref_pdb = f"../prepared_pdbs/filtered_prolif_ready_{pdb_code.lower()}.pdb"

    if not os.path.exists(sim_dir):
        return f"[Skipping] {pdb_code}: Sim directory not found ({sim_dir})"

    try:
        analyser = HemeAnalysis(
            simulation_path=sim_dir,
            ligand_id="LIG",
            reference_pdb_path=ref_pdb,
            reference_ligand_id=lig_code,
            docked_pdb_path=docked_pdb,
            docked_ligand_id="LIG" # or lig_code based on your preference
        )

        results = analyser.run_all_analyses()

        if results:
            return {'status': 'success', 'data': results, 'pdb': pdb_code}
        else:
            return f"[Failed] {pdb_code}: run_all_analyses returned None"

    except Exception as e:
        return f"[Error] {pdb_code}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Run batch Heme analysis.")
    parser.add_argument("filename", help="Path to the input CSV file.")
    parser.add_argument(
        "--cores", 
        type=int, 
        default=1, 
        help="Number of CPU cores to use. Defaults to one."
    )

    args = parser.parse_args()
    filename = args.filename
    num_cores = args.cores
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found.")
        sys.exit(1)

    print(f"Reading input CSV: {filename}")
    df = pd.read_csv(filename)
    if num_cores:
        print(f"Using {num_cores} cores.")
    else:
        print(f"Using default (all available) cores: {os.cpu_count()}")
    tasks = []
    for index, row in df.iterrows():
        task = (
            index,
            row.get('file location'),
            row.get('pdb code'),
            row.get('ligand code')
        )
        tasks.append(task)

    print(f"Found {len(tasks)} simulations to process.")
    
    successful_results = []
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results_iterator = executor.map(process_single_simulation, tasks)

        for result in results_iterator:
            if isinstance(result, dict) and result.get('status') == 'success':
                successful_results.append(result['data'])
                print(f"  + Finished: {result['pdb']}")
            else:
                print(result)

    end_time = time.time()
    print(f"\nParallel processing complete in {end_time - start_time:.2f} seconds.")

    print(f"Total successful simulations: {len(successful_results)}")

    if len(successful_results) > 0:
        print(f"Starting aggregation for {len(successful_results)} runs...")

        
        agg_ref_file = filename.replace(".csv", "_stats_ref.csv")
        aggregate_ligand_statistics(
            successful_results,
            target_diff_type='diff_to_ref',
            output_csv=agg_ref_file
        )

        agg_dock_file = filename.replace(".csv", "_stats_docked.csv")
        aggregate_ligand_statistics(
            successful_results,
            target_diff_type='diff_to_docked',
            output_csv=agg_dock_file
        )

        print("\nBatch analysis and aggregation complete.")
    else:
        print("\nNo successful analyses to aggregate.")

if __name__ == "__main__":
    main()
