#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import argparse

# PLIP XML tags usually look like this. We map them to cleaner column names.
INTERACTION_TYPES = {
    'hydrophobic_interaction': 'Hydrophobic',
    'hydrogen_bond': 'HBond',
    'water_bridge': 'WaterBridge',
    'salt_bridge': 'SaltBridge',
    'pi_stack': 'PiStack',
    'pi_cation_interaction': 'PiCation',
    'halogen_bond': 'Halogen',
    'metal_complex': 'Metal'
}

# Mapping PLIP XML plural group tags to singular interaction names
NORMALIZE_TYPES = {
    'hydrophobic_interactions': 'hydrophobic_interaction',
    'hydrogen_bonds': 'hydrogen_bond',
    'water_bridges': 'water_bridge',
    'salt_bridges': 'salt_bridge',
    'pi_stacks': 'pi_stack',
    'pi_cation_interactions': 'pi_cation_interaction',
    'halogen_bonds': 'halogen_bond',
    'metal_complexes': 'metal_complex'
}

def normalize_type(type_name):
    """
    Normalizes a PLIP interaction type string to its singular form.
    E.g. 'hydrophobic_interactions' -> 'hydrophobic_interaction'
    """
    t = str(type_name).strip()
    return NORMALIZE_TYPES.get(t, t)

def parse_interaction_string(details_str):
    """
    Parses the string "RES A-123 : type; RES A-124 : type"
    Returns a set of tuples: {('RES A-123', 'type'), ...}
    """
    if pd.isna(details_str) or str(details_str).strip() == "":
        return set()

    interactions = set()
    items = str(details_str).split(";")
    for item in items:
        if ":" in item:
            parts = item.split(":")
            res_info = parts[0].strip()  # e.g., "PHE A-212"
            raw_type = parts[1].strip()  # e.g., "hydrophobic_interaction"
            
            # Normalize to ensure matching (Singular vs Plural)
            int_type = normalize_type(raw_type)
            
            interactions.add((res_info, int_type))

    return interactions

def parse_xml_interactions(xml_path, target_ligand, heuristic_residues=None):
    """
    Parses a PLIP XML file with STRICT filtering for the target_ligand.

    1. Filter sites: Only consider interactions where 'restype_lig' == target_ligand.
    2. Heuristic: If multiple sites match the target ligand (e.g. Chain A vs Chain B),
       select the one with the most residue overlap with the MD simulation.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"  [XML Error] Could not parse {os.path.basename(xml_path)}: {e}")
        return set()

    candidate_sites = []

    # Iterate over all binding sites
    for site in root.findall('bindingsite'):
        interaction_group = site.find('interactions')
        if interaction_group is None:
            continue

        current_site_interactions = set()
        current_site_residues = set()
        has_correct_ligand = False

        for int_type_group in interaction_group:
            # XML Tag is usually plural (e.g., 'hydrophobic_interactions')
            raw_group_tag = int_type_group.tag
            
            # Convert to singular for consistency with CSV and INTERACTION_TYPES keys
            interaction_type_name = normalize_type(raw_group_tag)

            for interaction in int_type_group:
                # STRICT FILTER: Check ligand name
                lig_name = interaction.find('restype_lig').text

                # Skip if this specific interaction is not with our target ligand
                # This handles composite sites or just ensures we are looking at the right thing
                if lig_name != target_ligand:
                    continue

                has_correct_ligand = True

                resnr = interaction.find('resnr').text
                reschain = interaction.find('reschain').text
                restype = interaction.find('restype').text

                # Format: "RES CA-123"
                # Note: Adjusting format to match common PLIP outputs like "TYR A-123" if needed,
                # but staying consistent with the keys used in parse_interaction_string
                res_full = f"{restype} {reschain}-{resnr}"

                sig_tuple = (res_full, interaction_type_name)
                current_site_interactions.add(sig_tuple)
                current_site_residues.add(res_full)

        # Only consider this site if it actually contained our target ligand
        if has_correct_ligand and current_site_interactions:
            candidate_sites.append({
                'interactions': current_site_interactions,
                'residues': current_site_residues
            })

    if not candidate_sites:
        print(f"  [Warning] No binding sites found for ligand '{target_ligand}' in {os.path.basename(xml_path)}")
        return set()

    # If only one site matches the ligand, return it
    if len(candidate_sites) == 1:
        return candidate_sites[0]['interactions']

    # If multiple sites match the ligand (e.g. multimer), use heuristic to pick the one simulated
    best_interactions = set()
    max_overlap = -1

    if heuristic_residues:
        for site in candidate_sites:
            # We look for overlap between the XML site residues and what was observed in the MD CSV
            overlap = len(site['residues'].intersection(heuristic_residues))
            if overlap > max_overlap:
                max_overlap = overlap
                best_interactions = site['interactions']
    else:
        # Fallback: take the first matching site
        best_interactions = candidate_sites[0]['interactions']

    return best_interactions

def process_simulation_csv(csv_path, folder_path, pdb_id, target_ligand):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

    # 1. Gather Heuristic Data (Conserved residues from MD)
    # This helps us identify WHICH binding site (Chain A vs B) is being tracked in the CSV
    md_observed_residues = set()
    if 'Conserved_Details' in df.columns:
        for details in df['Conserved_Details']:
            parsed = parse_interaction_string(details)
            for res_name, _ in parsed:
                md_observed_residues.add(res_name)

    # 2. Find and Parse Reference XML
    candidates = [
        f"{pdb_id.upper()}_report.xml",
        f"{pdb_id.upper()}_aligned_report.xml",
        f"{pdb_id.lower()}_report.xml"
    ]

    chosen_xml = None
    for c in candidates:
        full_p = os.path.join(folder_path, c)
        if os.path.exists(full_p):
            chosen_xml = full_p
            break

    reference_set = set()

    if chosen_xml:
        # NOW PASSING target_ligand FOR STRICT FILTERING
        reference_set = parse_xml_interactions(chosen_xml, target_ligand, heuristic_residues=md_observed_residues)
        if len(reference_set) == 0:
             # Skip if XML parsing returned nothing (no reference interactions found for this ligand)
             print(f"  [Info] XML found but no reference interactions for '{target_ligand}'. Skipping system.")
             return None
    else:
        # Skip if XML file is missing entirely
        print(f"  [Warning] Reference XML not found in {os.path.basename(folder_path)}. Skipping system.")
        return None

    # 3. Process Frames
    processed_rows = []
    total_ref = len(reference_set)

    for index, row in df.iterrows():
        conserved_set = parse_interaction_string(row.get('Conserved_Details', ''))

        # Calculate lost
        lost_set = reference_set - conserved_set

        conserved_count = len(conserved_set)
        lost_count = len(lost_set)

        fraction_retained = conserved_count / total_ref if total_ref > 0 else 0.0

        row_data = {
            'Frame': row['Frame'],
            'Fraction_Retained': fraction_retained,
            'Total_Conserved': conserved_count,
            'Total_Lost': lost_count,
            'Conserved_Residues': "; ".join([x[0] for x in conserved_set]),
            'Lost_Residues': "; ".join([x[0] for x in lost_set])
        }

        for xml_tag, col_name in INTERACTION_TYPES.items():
            # xml_tag here is Singular (key of INTERACTION_TYPES). 
            # Our sets (conserved_set, lost_set) are now guaranteed to be Singular due to normalization.
            row_data[f'{col_name}_Conserved'] = sum(1 for res, type_ in conserved_set if type_ == xml_tag)
            row_data[f'{col_name}_Lost'] = sum(1 for res, type_ in lost_set if type_ == xml_tag)

        processed_rows.append(row_data)

    return pd.DataFrame(processed_rows)

def main():
    parser = argparse.ArgumentParser(description="Aggregate PLIP CSVs with proper Ligand filtering.")
    parser.add_argument("--dir", required=True, help="Main directory containing subfolders *-{PDB}")
    parser.add_argument("--csv", required=True, help="Path to ligand_descriptors.csv (to lookup LigID)")
    parser.add_argument("--out", default="aggregated_plip_results.csv", help="Output filename")

    args = parser.parse_args()

    # 1. Load CSV Map
    try:
        df_map = pd.read_csv(args.csv)
        pdb_ligand_map = {}
        for _, row in df_map.iterrows():
            # Assuming columns are 'PDB ID' and 'LigID' based on description
            p_id = str(row['PDB ID']).strip().upper()
            l_id = str(row['LigID']).strip()
            pdb_ligand_map[p_id] = l_id
        print(f"Loaded {len(pdb_ligand_map)} ligand mappings.")
    except Exception as e:
        print(f"Error loading CSV map: {e}")
        return

    # 2. Find files
    search_pattern = os.path.join(args.dir, "*", "plip_md_results.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"No 'plip_md_results.csv' files found in {args.dir}")
        return

    print(f"Found {len(csv_files)} simulation files. Aggregating...")

    all_data = []

    for csv_file in csv_files:
        folder_path = os.path.dirname(csv_file)
        parent_folder_name = os.path.basename(folder_path)

        # Robust PDB extraction
        # Assumes folder format like "system-1abc" or just "1abc"
        pdb_id = parent_folder_name.split("-")[-1].upper() if "-" in parent_folder_name else parent_folder_name.upper()

        # Lookup Ligand
        target_ligand = pdb_ligand_map.get(pdb_id)

        if not target_ligand:
            print(f"Skipping {pdb_id}: Ligand ID not found in mapping CSV.")
            continue

        print(f"Processing {pdb_id} (Ligand: {target_ligand})...", end="\r")

        df_processed = process_simulation_csv(csv_file, folder_path, pdb_id, target_ligand)
        if df_processed is not None and not df_processed.empty:
            df_processed.insert(0, 'PDB_ID', pdb_id)
            all_data.append(df_processed)

    print("\nMerging data...")
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)

        # Sort/Reorder columns
        cols = ['PDB_ID', 'Frame', 'Fraction_Retained', 'Total_Conserved', 'Total_Lost']
        type_cols = []
        for col_name in INTERACTION_TYPES.values():
            type_cols.append(f'{col_name}_Conserved')
            type_cols.append(f'{col_name}_Lost')

        final_cols = cols + type_cols + ['Conserved_Residues', 'Lost_Residues']
        # Filter to only columns that actually exist in the dataframe
        existing_cols = [c for c in final_cols if c in final_df.columns]
        final_df = final_df[existing_cols]

        final_df.to_csv(args.out, index=False)
        print(f"Done! Saved to {args.out} (Rows: {len(final_df)})")
    else:
        print("No valid data found.")

if __name__ == "__main__":
    main()
