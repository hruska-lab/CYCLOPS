import pytest
from pathlib import Path
from cyclops.cyclops import run_heme_simulation

def test_run_quick_simulation(tmp_path):
    """
    Runs a short (0.1 ns) simulation to test the full pipeline.
    
    This test uses the 'tmp_path' fixture from pytest to create a
    temporary directory for all output files, which is automatically
    cleaned up after the test.
    """
    
    # --- Test Parameters ---
    
    # 1. Ligand Name (the 3-letter code in the PDB) in real life it should be 2QH
    ligand_name = "UNL"
    
    # 2. Ligand SMILES string
    # 2QH
    ligand_smiles = "Cc1cc(ccc1OC)[C@]2(C3=NCC(CN3C(=N2)N)(F)F)c4ccc(c(c4)c5cccnc5)F"
    
    # 3. Simulation length
    simulation_time_ns = 0.1
    
    # 4. Set paths
    # tmp_path is a 'pathlib.Path' object pointing to a temporary directory
    output_dir = str(tmp_path) 
    
    # Get the path to the PDB file *relative to this test file*
    # Path.cwd() / "cyclops" / "tests" / "1W0E-4NY4.pdb"
    # A more robust way:
    test_dir = Path(__file__).parent
    pdb_file = test_dir / "1W0E-4NY4.pdb"
    
    # Check if the PDB file exists before running
    assert pdb_file.exists(), f"PDB file not found at {pdb_file}"
    
    print(f"\n--- Starting Quick Simulation Test ---")
    print(f"PDB Input: {pdb_file}")
    print(f"Output Dir: {output_dir}")
    print(f"Ligand: {ligand_name}")
    print(f"----------------------------------------")
    
    # --- Run the Simulation ---
    run_heme_simulation(
        pdb_file_path=str(pdb_file),
        ligand_smiles=ligand_smiles,
        ligand_name=ligand_name,
        output_dir=output_dir,
        n_nanoseconds=simulation_time_ns,
        verbose=True  # Set to True to see the simulation output
    )
    
    # --- Verify Output Files ---
    # Check that the most important final files were created
    output_path = Path(output_dir)
    
    # The PDB topology file for the final, solvated system
    topology_file = output_path / "topology_complex.pdb"
    assert topology_file.exists(), "Final topology file was not created."
    
    # The simulation trajectory
    trajectory_file = output_path / "trajectory_complex.xtc"
    assert trajectory_file.exists(), "Trajectory file was not created."

    # The simulation log
    log_file = output_path / "log_E_temp_vol.txt"
    assert log_file.exists(), "Simulation log file was not created."
    
    print(f"\n--- Test Complete: All output files found. ---")