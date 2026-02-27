import mdtraj as md
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# A helper function to create the plots
def _plot_stability_logs(simulation_path, data, prefix):
    """Generates stability plots for energy, temp, and volume."""
    step = data[:, 0]
    potential_energy = data[:, 1]
    temperature = data[:, 2]
    volume = data[:, 3]

    plt.figure()
    plt.plot(step, potential_energy)
    plt.xlabel("Step")
    plt.ylabel(r"Potential Energy $\frac{kJ}{mol}$")
    plt.savefig(f'{simulation_path}/{prefix}_energy_plot.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    plt.figure()
    plt.plot(step, temperature)
    plt.xlabel("Step")
    plt.ylabel("Temperature [K]")
    plt.yscale('log')
    plt.savefig(f'{simulation_path}/{prefix}_temperature_plot.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    plt.figure()
    plt.plot(step, volume)
    plt.xlabel("Step")
    plt.ylabel("Volume [nm$^3$]")
    plt.yscale('log')
    plt.savefig(f'{simulation_path}/{prefix}_volume_plot.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


class HemeAnalysis:
    """
    A comprehensive analysis suite for Heme-Ligand Molecular Dynamics simulations.

    This class processes simulation outputs to generate:
    
    1. **Stability Plots:** Energy, Temperature, and Volume over time.
    2. **Heme Geometry:** Tracking Fe-N and Fe-S distances and angles to ensure the 
       force field constraints held the heme in the correct geometry.
    3. **Ligand Stability:** Calculating the distance between the Ligand and the Heme Iron 
       frame-by-frame.
    4. **Comparisons:** If provided, compares the simulation trajectory against 
       Experimental (Reference) or Docked structures.

    Parameters
    ----------
    simulation_path : str
        Path to the directory containing the simulation outputs (trajectory, topology, logs).
    ligand_id : str
        The 3-letter residue name of the ligand in the simulation topology.
    reference_pdb_path : str, optional
        Path to a reference PDB (e.g., Crystal structure) for comparison.
    reference_ligand_id : str, optional
        The 3-letter residue name of the ligand in the reference PDB. 
        Defaults to `ligand_id` if not set.
    docked_pdb_path : str, optional
        Path to a docked pose PDB for comparison.
    docked_ligand_id : str, optional
        The 3-letter residue name of the ligand in the docked PDB. 
        Defaults to checking 'UNL' or 'LIG' if not specified.
    """
    
    def __init__(self, 
                 simulation_path: str, 
                 ligand_id: str,
                 reference_pdb_path: str = None,
                 reference_ligand_id: str = None,
                 docked_pdb_path: str = None,
                 docked_ligand_id: str = None):
        
        self.sim_path = Path(simulation_path)
        self.sim_name = self.sim_path.name

        # --- File Paths ---
        self.xtc_file = self.sim_path / "trajectory_complex.xtc"
        self.top_file = self.sim_path / "topology_complex.pdb" 
        self.log_file = self.sim_path / "log_E_temp_vol.txt"
        self.initial_coords_file = self.sim_path / "packed_complex.pdb"
        
        # --- Handle Ligand IDs ---
        self.ligand_id = ligand_id
        self.ref_ligand_id = reference_ligand_id if reference_ligand_id else ligand_id
        self.docked_ligand_id = docked_ligand_id 
        
        # --- Handle optional reference files ---
        self.ref_pdb = Path(reference_pdb_path) if reference_pdb_path else None
        self.docked_pdb = Path(docked_pdb_path) if docked_pdb_path else None
        
        self.has_reference = self.ref_pdb is not None and self.ref_pdb.exists()
        self.has_docked = self.docked_pdb is not None and self.docked_pdb.exists()

        # --- Storage for loaded data ---
        self.traj = None
        self.traj_ref = None
        self.traj_dock = None
        self.log_data = None
        
        self.atoms = {}
        self.results = {}

    def _check_simulation(self) -> bool:
        """Checks if the simulation output files exist and are not empty."""
        if not self.xtc_file.exists() or os.path.getsize(self.xtc_file) < 100:
            print(f"Warning: XTC file missing or empty for {self.sim_name}")
            return False
        if not self.top_file.exists():
            print(f"Warning: Topology file missing for {self.sim_name}")
            return False
        if not self.initial_coords_file.exists():
            print(f"Warning: Initial coords file (packed_complex.pdb) missing for {self.sim_name}")
            return False
            
        if not self.log_file.exists():
            print(f"Warning: Log file missing for {self.sim_name}")
            return False
        return True

    def _load_data(self):
        """Loads all trajectories and log data into memory."""
        print(f"Loading data for {self.sim_name}...")
        self.traj = md.load_xtc(str(self.xtc_file), top=str(self.top_file))
        self.log_data = np.loadtxt(str(self.log_file), delimiter=',')
        
        if self.has_reference:
            print(f"  Loading reference: {self.ref_pdb.name}")
            self.traj_ref = md.load(str(self.ref_pdb))
        if self.has_docked:
            print(f"  Loading docked: {self.docked_pdb.name}")
            self.traj_dock = md.load(str(self.docked_pdb))
            
    def _select_atoms(self):
        """Finds and stores all relevant atom indices."""
        print("Selecting atoms...")
        
        topo = self.traj.topology
        self.atoms['fe_main'] = topo.select('element Fe')
        self.atoms['n1_main'] = topo.select("resname HEM and name N1x")
        self.atoms['n2_main'] = topo.select("resname HEM and name N2x")
        self.atoms['n3_main'] = topo.select("resname HEM and name N3x")
        self.atoms['n4_main'] = topo.select("resname HEM and name N4x")
        
        s_cys_idx = topo.select("resname DYE and element S")
        
        if len(s_cys_idx) > 0 and len(self.atoms['fe_main']) > 0:
            distances = md.compute_distances(self.traj[0], atom_pairs=list(zip(s_cys_idx, [self.atoms['fe_main'][0]]*len(s_cys_idx))))
            closest_s_index = s_cys_idx[np.argmin(distances)]
            self.atoms['s_main'] = np.array([closest_s_index])
        else:
            self.atoms['s_main'] = []

        # Robust ligand selection
        self.atoms['ligand_main'] = topo.select(f'resname {self.ligand_id} and not symbol H')

        # --- Reference Atoms ---
        if self.has_reference:
            topo_ref = self.traj_ref.topology
            self.atoms['fe_ref'] = topo_ref.select('element Fe')
            self.atoms['ligand_ref'] = topo_ref.select(f'resname {self.ref_ligand_id} and not symbol H')

        # --- Docked Atoms ---
        if self.has_docked:
            topo_dock = self.traj_dock.topology
            self.atoms['fe_dock'] = topo_dock.select("element Fe")
            
            selection_string = ""
            if self.docked_ligand_id:
                selection = topo_dock.select(f'resname {self.docked_ligand_id} and not symbol H')
                if len(selection) > 0:
                    selection_string = f'resname {self.docked_ligand_id} and not symbol H'
            
            if not selection_string:
                for fallback_id in ['UNL', 'LIG']:
                    selection = topo_dock.select(f'resname {fallback_id} and not symbol H')
                    if len(selection) > 0:
                        selection_string = f'resname {fallback_id} and not symbol H'
                        break
            
            if not selection_string:
                self.atoms['ligand_dock'] = np.array([])
            else:
                self.atoms['ligand_dock'] = topo_dock.select(selection_string)

    def run_all_analyses(self):
        """Runs the full analysis pipeline."""
        if not self._check_simulation():
            return None
            
        self._load_data()
        self._select_atoms()
        
        print("Calculating metrics...")
        self.results['stability'] = self.calculate_stability_metrics()
        self.results['heme_coords'] = self.calculate_heme_coords()
        self.results['ligand_fe'] = self.calculate_ligand_fe_distance()
        
        # Calculate differences (dropping 10% equilibration)
        self.results['ligand_fe_diff'] = self.calculate_ligand_fe_differences(drop_fraction=0.10)
        
        print("Generating plots and saving data...")
        self.plot_stability()
        self.plot_heme_coords()
        self.plot_ligand_fe_distance()
        self.plot_difference_histogram()
        self.save_diff_to_csv() 
        
        print(f"\nAnalysis complete for {self.sim_name}")
        return self.results

    def calculate_stability_metrics(self):
        """Returns the mean values from the log file."""
        return {
            "mean_energy": np.mean(self.log_data[100:, 1]), 
            "mean_temp": np.mean(self.log_data[100:, 2]),
            "mean_volume": np.mean(self.log_data[100:, 3]),
        }

    def calculate_heme_coords(self):
        """Calculates heme coordination distances and angles over time."""
        if len(self.atoms['fe_main']) == 0: return {}

        n_fe_pairs = [
            [self.atoms['n1_main'][0], self.atoms['fe_main'][0]],
            [self.atoms['n2_main'][0], self.atoms['fe_main'][0]],
            [self.atoms['n3_main'][0], self.atoms['fe_main'][0]],
            [self.atoms['n4_main'][0], self.atoms['fe_main'][0]],
        ]
        
        results = {}
        dists_n_fe = md.compute_distances(self.traj, n_fe_pairs)
        results["n_fe_dists"] = dists_n_fe
        
        if len(self.atoms['s_main']) > 0:
            s_fe_pair = [[self.atoms['s_main'][0], self.atoms['fe_main'][0]]]
            dist_s_fe = md.compute_distances(self.traj, s_fe_pair)
            results["s_fe_dist"] = dist_s_fe
        
        # Angles
        angle_14_idx = [[self.atoms['n1_main'][0], self.atoms['fe_main'][0], self.atoms['n4_main'][0]]]
        angle_23_idx = [[self.atoms['n2_main'][0], self.atoms['fe_main'][0], self.atoms['n3_main'][0]]]
        
        results["angle_14"] = np.degrees(md.compute_angles(self.traj, angle_14_idx))
        results["angle_23"] = np.degrees(md.compute_angles(self.traj, angle_23_idx))
        
        # Get initial/restraint values
        try:
            initial_top = md.load(self.initial_coords_file)
            results["r_n_fe_dists"] = md.compute_distances(initial_top, n_fe_pairs)[0]
            if len(self.atoms['s_main']) > 0:
                results["r_s_fe_dist"] = md.compute_distances(initial_top, s_fe_pair)[0]
            results["r_angle_14"] = np.degrees(md.compute_angles(initial_top, angle_14_idx))[0]
            results["r_angle_23"] = np.degrees(md.compute_angles(initial_top, angle_23_idx))[0]
        except Exception:
            pass

        return results

    def calculate_ligand_fe_distance(self):
        """Calculates the closest ligand atom to Fe distance."""
        data = {"sim": None, "ref": None, "docked": None, "initial": None}

        if len(self.atoms['ligand_main']) == 0 or len(self.atoms['fe_main']) == 0:
            return data
            
        dist_time = md.compute_distances(self.traj, [[a, self.atoms['fe_main'][0]] for a in self.atoms['ligand_main']])
        data['sim'] = np.min(dist_time, axis=1)
       
        try:
            initial_top = md.load(self.initial_coords_file)
            dist_initial = md.compute_distances(initial_top, [[a, self.atoms['fe_main'][0]] for a in self.atoms['ligand_main']])
            data['initial'] = np.min(dist_initial, axis=1)[0]
        except Exception:
            pass
        
        if self.has_reference:
            if len(self.atoms['ligand_ref']) > 0 and len(self.atoms['fe_ref']) > 0:
                dist_ref = md.compute_distances(self.traj_ref, [[a, self.atoms['fe_ref'][0]] for a in self.atoms['ligand_ref']])
                data['ref'] = np.min(dist_ref, axis=1)[0]

        if self.has_docked:
            if len(self.atoms['ligand_dock']) > 0 and len(self.atoms['fe_dock']) > 0:
                dist_dock = md.compute_distances(self.traj_dock, [[a, self.atoms['fe_dock'][0]] for a in self.atoms['ligand_dock']])
                data['docked'] = np.min(dist_dock, axis=1)[0]
            
        return data

    def calculate_ligand_fe_differences(self, drop_fraction=0.10):
        """
        Calculates the frame-by-frame difference between simulation Lig-Fe distance
        and the reference/docked distances, dropping the initial portion (equilibration).
        
        Parameters
        ----------
        drop_fraction : float
            Fraction of the beginning of the trajectory to drop (e.g., 0.10 for 10%).
            
        Returns
        -------
        dict: Keys 'diff_to_ref', 'diff_to_docked', 'start_frame'.
        """
        data = self.results.get('ligand_fe')
        if not data or data['sim'] is None:
            return {}

        diffs = {}
        n_frames = len(data['sim'])
        start_idx = int(n_frames * drop_fraction)
        
        sim_sliced = data['sim'][start_idx:]
        diffs['start_frame'] = start_idx
        
        # Difference to Reference (Crystal)
        if data['ref'] is not None:
            diffs['diff_to_ref'] = np.abs(sim_sliced - data['ref'])
            
        # Difference to Docking
        if data['docked'] is not None:
            diffs['diff_to_docked'] = np.abs(sim_sliced - data['docked'])
            
        return diffs

    def plot_stability(self):
        _plot_stability_logs(str(self.sim_path), self.log_data, self.sim_name)

    def plot_heme_coords(self):
        data = self.results['heme_coords']
        if not data: return

        plt.figure()
        plt.plot(data['n_fe_dists'][:, 0], label='N1-Fe')
        plt.plot(data['n_fe_dists'][:, 1], label='N2-Fe')
        plt.plot(data['n_fe_dists'][:, 2], label='N3-Fe')
        plt.plot(data['n_fe_dists'][:, 3], label="N4-Fe")
        if 's_fe_dist' in data:
            plt.plot(data['s_fe_dist'], label="S(Cys)-Fe")
        
        if 'r_n_fe_dists' in data:
            plt.axhline(y=data['r_n_fe_dists'][0], color='C0', linestyle='--', label='N1-Fe (initial)')
        
        plt.title('Heme Coordination Distances')
        plt.xlabel('Frame')
        plt.ylabel('Distance [nm]')
        plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))
        plt.savefig(self.sim_path / f'{self.sim_name}_heme_distances.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_ligand_fe_distance(self):
        data = self.results['ligand_fe']
        if data['sim'] is None: return

        plt.figure()
        plt.plot(data['sim'], label='Ligand - Fe (sim)')
        #if data['initial'] is not None:
        #        plt.axhline(y=data['initial'], color='green', linestyle='--', label='Initial (t=0)')
        if data['docked'] is not None:
            plt.axhline(y=data['docked'], color='red', linestyle='--', label='Docked')
        if data['ref'] is not None:
            plt.axhline(y=data['ref'], color='blue', linestyle='--', label='Reference')

        plt.title('Closest Ligand-Fe Distance')
        plt.xlabel('Frame')
        plt.ylabel('Distance [nm]')
        plt.legend()
        plt.savefig(self.sim_path / f'{self.sim_name}_ligand_fe_distance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_difference_histogram(self):
        """
        Plots histograms of the calculated differences (Sim vs Ref/Docked).
        Uses the data already processed (and clipped) by calculate_ligand_fe_differences.
        """
        diffs = self.results.get('ligand_fe_diff')
        if not diffs: return

        for key in ['diff_to_ref', 'diff_to_docked']:
            if key not in diffs: continue
            
            values = diffs[key]
            if len(values) == 0: continue

            mean_val = np.mean(values)
            std_val = np.std(values)
            
            plt.figure()
            plt.hist(values, bins=50, alpha=0.7, edgecolor='black')
            plt.title(f'{key} (After dropping first 10%)\nMean: {mean_val:.3f}, Std: {std_val:.3f}')
            plt.xlabel('Difference [nm]')
            plt.ylabel('Count')
            plt.savefig(self.sim_path / f'{self.sim_name}_{key}_hist.png', dpi=300, bbox_inches='tight')
            plt.close()

    def save_diff_to_csv(self):
        """Saves the frame-by-frame differences to a CSV file."""
        diffs = self.results.get('ligand_fe_diff')
        if not diffs: return

        # Prepare DataFrame
        df_dict = {}
        
        length = 0
        start_frame = diffs.get('start_frame', 0)
        
        if 'diff_to_ref' in diffs:
            df_dict['Diff_to_Ref'] = diffs['diff_to_ref']
            length = len(diffs['diff_to_ref'])
        
        if 'diff_to_docked' in diffs:
            df_dict['Diff_to_Docked'] = diffs['diff_to_docked']
            length = len(diffs['diff_to_docked'])
            
        if length > 0:
            # Create Frame index column matching trajectory frames
            df_dict['Frame'] = np.arange(start_frame, start_frame + length)
            
            df = pd.DataFrame(df_dict)
            # Reorder to put Frame first
            cols = ['Frame'] + [c for c in df.columns if c != 'Frame']
            df = df[cols]
            
            save_file = self.sim_path / f'{self.sim_name}_ligand_fe_diff.csv'
            df.to_csv(save_file, index=False)
            print(f"  Saved difference data to {save_file.name}")

def _calc_array_stats(arr):
    if len(arr) == 0: return {}
    q75, q25 = np.percentile(arr, [75 ,25])
    return {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "iqr": q75 - q25,
        "q25": q25,
        "q75": q75,
        "count": len(arr)
    }

def aggregate_ligand_statistics(analysis_input, 
                                target_diff_type='diff_to_ref', 
                                output_csv=None):
    """
    Aggregates and statistically analyzes Ligand-Fe distance deviations across simulations.

    This function takes results from `HemeAnalysis`, pools the frame-by-frame 
    differences (e.g., Simulation vs Crystal Structure), and computes statistics.
    It splits the data into two halves ('part1' and 'part2') to check for convergence 
    or drift during the simulation.

    Parameters
    ----------
    analysis_input : HemeAnalysis, dict, or list
        The source data. Can be a single `HemeAnalysis` object, its `.results` dictionary, 
        or a list of these objects/dictionaries from multiple replicas.
    target_diff_type : str, optional
        The metric to analyze. Options:
        - ``'diff_to_ref'``: Difference between Sim and Crystal structure.
        - ``'diff_to_docked'``: Difference between Sim and Docked pose.
    output_csv : str, optional
        If provided, saves the flattened statistical summary (Mean, Std, IQR) to this file path.

    Returns
    -------
    dict
        A dictionary containing statistics (Mean, Std, IQR, Quartiles) for:
        - ``'total'``: The full dataset.
        - ``'part1'``: The first half of the frames.
        - ``'part2'``: The second half of the frames.
    """
    
    # Normalize input to a list of items
    if isinstance(analysis_input, (dict, HemeAnalysis)):
        items = [analysis_input]
    elif isinstance(analysis_input, list):
        items = analysis_input
    else:
        print(f"Invalid input type: {type(analysis_input)}")
        return None

    pooled_diffs = []
    
    for item in items:
        # Extract results dict if it is an object
        results = None
        if isinstance(item, HemeAnalysis):
            if hasattr(item, 'results'):
                results = item.results
        elif isinstance(item, dict):
            results = item
            
        if results and 'ligand_fe_diff' in results:
            diff_dict = results['ligand_fe_diff']
            if target_diff_type in diff_dict:
                pooled_diffs.append(diff_dict[target_diff_type])
            
    if not pooled_diffs:
        print(f"No data found for aggregation type: {target_diff_type}")
        return None
        
    # Concatenate all runs into one large array
    all_data = np.concatenate(pooled_diffs)
    
    # Split into halves
    mid_point = len(all_data) // 2
    part1 = all_data[:mid_point]
    part2 = all_data[mid_point:]
    
    # Calculate statistics for each segment
    stats_all = _calc_array_stats(all_data)
    stats_p1 = _calc_array_stats(part1)
    stats_p2 = _calc_array_stats(part2)
    
    # Combine into a single structure
    results = {
        "type": target_diff_type,
        "total": stats_all,
        "part1": stats_p1,
        "part2": stats_p2
    }
    
    # Flatten for CSV if needed (one row with prefixed columns)
    if output_csv:
        try:
            flat_dict = {"type": target_diff_type}
            for k, v in stats_all.items(): flat_dict[f"total_{k}"] = v
            for k, v in stats_p1.items(): flat_dict[f"part1_{k}"] = v
            for k, v in stats_p2.items(): flat_dict[f"part2_{k}"] = v
            
            df = pd.DataFrame([flat_dict])
            df.to_csv(output_csv, index=False)
            print(f"Aggregated statistics saved to {output_csv}")
        except Exception as e:
            print(f"Failed to save aggregated stats to csv: {e}")

    return results