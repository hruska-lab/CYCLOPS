import os
import pandas as pd
import argparse
import sys

def process_directories(root_dir, output_prefix):
    """
    Traverses directories, finds specific CSVs based on subdir name, splits them, and aggregates data.
    """
    
    # Initialize storage for the 4 categories
    # We use lists to collect the data chunks, then concatenate them at the end (much faster)
    data_buckets = {
        'Diff_to_Ref_part_1': [],
        'Diff_to_Ref_part_2': [],
        'Diff_to_Docked_part_1': [],
        'Diff_to_Docked_part_2': []
    }

    files_found = 0

    print(f"Traversing '{root_dir}' looking for files matching '{{subdir}}_ligand_fe_diff.csv'...")

    # Walk through the directory tree
    for current_root, dirs, files in os.walk(root_dir):
        # Get the name of the current directory to construct the filename
        subdir_name = os.path.basename(current_root)
        target_filename = f"{subdir_name}_ligand_fe_diff.csv"

        if target_filename in files:
            file_path = os.path.join(current_root, target_filename)
            
            try:
                # Read the CSV
                df = pd.read_csv(file_path)
                
                # Check if required columns exist
                required_cols = ['Diff_to_Ref', 'Diff_to_Docked']
                if not all(col in df.columns for col in required_cols):
                    print(f"Skipping {file_path}: Missing required columns.")
                    continue

                # Drop 'Frame' if it exists (or just select the ones we want)
                # We simply ignore 'Frame' by selecting the other two explicitly
                
                # Calculate the split point (integer division)
                mid_point = len(df) // 2
                
                # Split the dataframe
                part_1 = df.iloc[:mid_point]
                part_2 = df.iloc[mid_point:]
                
                # Append data to respective buckets
                # We stick to the values to avoid index alignment issues during concatenation later
                data_buckets['Diff_to_Ref_part_1'].append(part_1['Diff_to_Ref'])
                data_buckets['Diff_to_Ref_part_2'].append(part_2['Diff_to_Ref'])
                
                data_buckets['Diff_to_Docked_part_1'].append(part_1['Diff_to_Docked'])
                data_buckets['Diff_to_Docked_part_2'].append(part_2['Diff_to_Docked'])
                
                files_found += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    if files_found == 0:
        print("No matching files found. Check your directory path.")
        return

    print(f"Found and processed {files_found} files. Generating outputs...")

    # Aggregate and Save
    for category, data_list in data_buckets.items():
        if data_list:
            # Concatenate all found series into one long series
            aggregated_series = pd.concat(data_list, ignore_index=True)
            
            # Construct filename: {output_argument}_{category}.csv
            # Example: my_run_Diff_to_Ref_part_1.csv
            output_filename = f"{output_prefix}_{category}.csv"
            
            # Save to CSV (header=True keeps the column name, index=False hides row numbers)
            aggregated_series.to_csv(output_filename, index=False, header=[category])
            print(f"Saved: {output_filename} ({len(aggregated_series)} rows)")

if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Recursively find, split, and aggregate CSV metrics.")
    
    parser.add_argument("root_dir", help="Path to the main directory to search")
    parser.add_argument("output_prefix", help="Prefix for the output files (e.g., 'experiment_1')")
    # Removed --target_file argument as naming is now dynamic based on directory

    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.root_dir):
        print(f"Error: Directory '{args.root_dir}' does not exist.")
        sys.exit(1)

    process_directories(args.root_dir, args.output_prefix)
