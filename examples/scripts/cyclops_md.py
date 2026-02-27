import pandas as pd
import sys
import os
from cyclops.cyclops import run_heme_simulation
from cyclops.rfaafixer import RFAAFixer

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_csv.py <path_to_csv_file>")
        sys.exit(1)

    filename = sys.argv[1]

    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found.")
        sys.exit(1)
    
    df_chunk = pd.read_csv(filename)

    for index, row in df_chunk.iterrows():
        file_loc = row.get('file location')
        pdb_code = row.get('pdb code')
        
        try:
            ligand = "LIG"
            smiles = row.get('smiles')

            pdb_file = os.path.join("pdb_inputs", file_loc)

            output_dir = os.path.join("outputs", file_loc[:9])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            fixed_pdb_path = os.path.join(output_dir, "rfaa_fixed.pdb")
            

            fixer = RFAAFixer(pdb_file)
            fixer.split_and_repair(fixed_pdb_path)

            run_heme_simulation(
                pdb_file_path=fixed_pdb_path, 
                ligand_smiles=smiles,
                ligand_name=ligand,
                output_dir=output_dir,
                n_nanoseconds=5,
                verbose=False
            )
            
        except Exception as e:
            print(f"FAILED: {file_loc}")
            print(f"Error details: {e}\n")

if __name__ == "__main__":
    main()
