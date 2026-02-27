import mdtraj
import openmm
import openmm.app
import openmm.unit
import sys
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
import urllib
from openff.interchange.drivers import (
    get_amber_energies,
    get_gromacs_energies,
    get_openmm_energies,
)
from openff.interchange.drivers.all import get_summary_data
from openbabel import pybel
from pdbfixer import PDBFixer
import os
from openff.toolkit.typing.engines.smirnoff import ForceField  
import time
from openmm import unit as openmm_unit
from openmm import MonteCarloBarostat
from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
from openmm.app import Modeller, ForceField as OMForceField, PDBFile
from openmm.unit import nanometer
from openff.interchange import Interchange

# Imports needed for PTM generation

from openff.pablo import (
    topology_from_pdb,
    STD_CCD_CACHE,
    ResidueDefinition,
    
)
from openff.pablo.chem import PEPTIDE_BOND
from openff.pablo.residue import BondDefinition, AtomDefinition


#internals

from . import utils 
from .data import get_data_file_path 
"""Provide the primary functions."""


def _prepare_inputs(
    pdb_file_path: str,
    ligand_smiles: str,
    ligand_name: str,
    simulation_dir: str,
    verbose: bool = False
) -> dict:
    """
    Takes the raw PDB and ligand info, splits all components,
    and prepares them (protonate, fix, etc.).
    Returns a dictionary of paths to the prepared files.
    """
    if verbose:
        print("Stage 1: Preparing all input files...")
        
    paths = {
        "complex": f"{simulation_dir}/complex.pdb",
        "ligand_unprot": f"{simulation_dir}/unprotonated_ligand.pdb",     
        "ligand_bonds_fixed": f"{simulation_dir}/ligand_neutral_fixed.sdf",  
        "ligand_final": f"{simulation_dir}/ligand_prepared_for_sim.sdf",
        "protein_plain": f"{simulation_dir}/plain_protein.pdb",
        "protein_fixed": f"{simulation_dir}/fixed_plain_protein.pdb",
        "heme_unprot": f"{simulation_dir}/heme_without_fe_unprotonated.pdb",
        "heme_prot_sdf": f"{simulation_dir}/protonated_heme_without_fe.sdf",
        "heme_prepared_sdf": f"{simulation_dir}/prepared_heme.sdf",
        "fe_pdb": f"{simulation_dir}/Fe.pdb",
        "fe_sdf": f"{simulation_dir}/Fe.sdf",
        "protein_ptm": f"{simulation_dir}/protein_with_CYF.pdb", 
    }
    SMILES_HEME = "C=Cc1c(C)c2cc5nc(cc4nc(cc3nc(cc1n2)c(C)c3C=C)c(C)c4CCC(=O)O)c(CCC(=O)O)c5C"
    # Load and save complex
    pdb_file = md.load(pdb_file_path)
    pdb_file.save(paths["complex"])

    # Extract and prepare ligand
    utils.extract_non_standard_residues(paths["complex"], simulation_dir, verbose)
    

    utils.prepare_ligand(
    paths["ligand_unprot"], 
    ligand_smiles,  
    paths["ligand_bonds_fixed"]
    )
    
    utils.protonate(
    paths["ligand_bonds_fixed"], 
    paths["ligand_final"], 
    polarOnly=False, 
    verbose=verbose,
    pH=7.4)
    
    # Extract and fix protein
    utils.extract_protein(paths["complex"], simulation_dir, verbose)
    utils.fix_protein(paths["protein_plain"], paths["protein_fixed"])

    # Extract and prepare Heme and Fe
    utils.extract_heme_fe(paths["complex"], simulation_dir, verbose)
    utils.protonate(paths["heme_unprot"], paths["heme_prot_sdf"], polarOnly=False, verbose=verbose)
    utils.prepare_heme(
        paths["heme_prot_sdf"], "HEM", SMILES_HEME, simulation_dir,
        output_sdf="prepared_heme.sdf"
    )

    utils.pdb_to_sdf(paths["fe_pdb"], paths["fe_sdf"])
    

    if verbose:
        print("Creating PTM protein PDB (CYS -> DYE)...")
    utils.create_ptm_protein_pdb(
        protein_pdb_path=paths["protein_fixed"],
        fe_pdb_path=paths["fe_pdb"],
        output_pdb_path=paths["protein_ptm"],
        new_residue_name="DYE",
        verbose=verbose
    )
    
    if verbose:
        print("...input preparation complete.")
    return paths

# --- Stage 2: Parameterize Components ---
def _parameterize_components(
    paths: dict,
    ligand_name: str,
    verbose: bool = False
) -> dict:
    """
    Loads prepared files and creates OpenFF Interchange objects
    for each component (protein+PTM, ligand, heme).
    """
    if verbose:
        print("Stage 2: Parameterizing components...")
    nagl_model_path = get_data_file_path("openff-gnn-am1bcc-1.0.0.pt")

    fe_ff_path = get_data_file_path("ff-Fe-2-Cl.offxml")

    if verbose:
        print("  Generating 'DYE' PTM residue definition...")

    cysteine = STD_CCD_CACHE["CYS"][0]
    cysteine_resdef = STD_CCD_CACHE["CYS"][0]
    Fe_atom = ResidueDefinition.anon_from_sdf(paths["fe_sdf"])
    thiol_Fe_click_smarts = (["[C:10]-[S:1]-[H:2]","[Fe:3]"],["[C:10]-[S:1]-[Fe:3]","[H:2]"])
    products = ResidueDefinition.react([cysteine, Fe_atom], thiol_Fe_click_smarts[0], thiol_Fe_click_smarts[1], product_residue_names = ["DYE"], product_linking_bonds = [PEPTIDE_BOND])
    dye = products[0][0] 
    Fe_atom_def = AtomDefinition("Fe1x", "Fe", "Fe", False, 1,False, None)
    CYF_atoms = list(dye.atoms[0:2]) + [Fe_atom_def] + list(dye.atoms[3:])
    sgfe_bond = BondDefinition("SG","Fe1x",1,False,None)
    CYF_Bonds = list(cysteine_resdef.bonds[0:11]) + [sgfe_bond] + list(cysteine_resdef.bonds[12:])
    new_dye = dye.replace(bonds=CYF_Bonds, atoms = CYF_atoms)
    
    if verbose:
        print("  'CYF' definition created.")

    # 2. Load the PTM-modified protein topology
    if verbose:
        print(f"  Loading PTM protein topology from {paths['protein_ptm']}...")
    protein_topology = topology_from_pdb(
        paths["protein_ptm"],
        residue_library =STD_CCD_CACHE.with_({"DYE": [new_dye]}))
    protein_top_no_water = utils.remove_crystal_water(protein_topology)
    
    # 3. Create the protein-PTM interchange
    # This FF list must contain the Fe parameters
    ff_protein_ptm = ForceField(
        "ff14sb_off_impropers_0.0.4.offxml", 
        "openff_unconstrained-2.2.1.offxml",
        fe_ff_path
    )
    
    protein_intrcg = Interchange.from_smirnoff(
        force_field=ff_protein_ptm,
        topology=protein_top_no_water
    )
    if verbose:
        print("  Protein-PTM interchange created.")

    if verbose:
        print("  Parameterizing Ligand...")
    ligand_openff = Molecule.from_file(paths["ligand_final"])
    NAGLToolkitWrapper().assign_partial_charges(ligand_openff, nagl_model_path)
    for atom in ligand_openff.atoms:
        atom.metadata["residue_name"] = ligand_name
    ligand_intrcg = Interchange.from_smirnoff(
        force_field=ForceField("openff_unconstrained-2.2.1.offxml"),
        topology=[ligand_openff],
        charge_from_molecules=[ligand_openff]
    )

    if verbose:
        print("  Parameterizing Heme (no Fe)...")
    heme_openff = Molecule.from_file(paths["heme_prepared_sdf"])
    NAGLToolkitWrapper().assign_partial_charges(heme_openff, nagl_model_path)
    for atom in heme_openff.atoms:
        atom.metadata["residue_name"] = "HEM"
    heme_intrcg = Interchange.from_smirnoff(
        force_field=ForceField("openff_unconstrained-2.2.1.offxml"),
        topology=[heme_openff],
        charge_from_molecules=[heme_openff]
    )



    
    if verbose:
        print("...parameterization complete.")
    return {
        "protein": protein_intrcg,
        "ligand": ligand_intrcg,
        "heme": heme_intrcg,
        "heme_mol": heme_openff, 
        "ligand_mol": ligand_openff
    }

# --- Stage 3: Assemble and Solvate ---
def _assemble_and_solvate(
    components: dict,
    simulation_dir: str,
    verbose: bool = False
) -> "openff.interchange.Interchange":
    """
    Combines component Interchanges, calculates charge,
    packs a box with water and counter-ions, and creates
    the final system Interchange object.
    """
    if verbose:
        print("Stage 3: Assembling and solvating system...")

    complex_intrcg = components["protein"].combine(components["ligand"])
    complex_intrcg = complex_intrcg.combine(components["heme"])

    charges = complex_intrcg["Electrostatics"].charges
    total_charge = sum(charge.m_as(unit.elementary_charge) for charge in charges.values())
    
            
           
    ions_to_add_dict = {}
    n_ions = 0
    ion = None
    if total_charge > 0:
        ions = int(round(total_charge))
        ion =  Molecule.from_smiles("[Cl-]")
    elif total_charge < 0:
        n_ions = int(round(abs(total_charge))) # Use absolute value
        ion =  Molecule.from_smiles("[Na+]")

    water = Molecule.from_mapped_smiles("[H:2][O:1][H:3]")
    for atom in water.atoms:
        atom.metadata["residue_name"] = "HOH"

    packed_topology = pack_box(
        molecules=[water, ion],
        number_of_copies=[23500, n_ions],
        solute=complex_intrcg.topology,
        box_shape=UNIT_CUBE,
        target_density=0.3 * unit.grams / (unit.centimeters**3),
    )
    packed_topology.to_file(f"{simulation_dir}/packed_complex.pdb")

    # Create final system interchange
    fe_ff_path = get_data_file_path("ff-Fe-2-Cl.offxml")
    sage_ff14sb = ForceField(
        "openff-2.2.1.offxml",
        "ff14sb_off_impropers_0.0.4.offxml",
        fe_ff_path,
    )
    
    final_interchange = sage_ff14sb.create_interchange(
        packed_topology,
        charge_from_molecules=[components["heme_mol"], components["ligand_mol"]]
    )

    if verbose:
        print("...assembly complete.")
    return final_interchange

# --- Stage 4: Add Custom Constraints ---
def _add_custom_constraints(
    simulation: "openmm.app.Simulation",
    packed_pdb_path: str,
    verbose: bool = False
) -> "openmm.app.Simulation":
    """
    Adds the specific HarmonicBond and HarmonicAngle forces
    for the Heme-Fe coordination.
    """
    if verbose:
        print("Stage 4: Adding custom Heme-Fe constraints (Fe-N bonds/angles)...")
    system = simulation.system
    topology = simulation.topology
    atoms = list(topology.atoms())

    nonbonded_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded_force = force
            break
    total_charge_2 = 0
    if nonbonded_force is None:
        print("Error: Could not find NonbondedForce in the OpenMM System.")
    new_charge_dye_sg = -0.3701 * openmm.unit.elementary_charge
    new_charge_dye_fe = +0.2620 * openmm.unit.elementary_charge
    if verbose:
        print("Modifying partial charges...")
    found_sg = False
    found_fe = False
    
    for i in range(system.getNumParticles()):
        atom = atoms[i]

        if atom.residue.name == 'DYE':
            if atom.name == 'SG':
                charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
                if verbose:
                    print(f"Found DYE-SG: Atom ID (Index) {i}")
                    print(f"  - Old charge: {charge.value_in_unit(openmm.unit.elementary_charge):+.6f} e")
                

                sigma_val = sigma.value_in_unit(openmm.unit.nanometer)
                epsilon_val = epsilon.value_in_unit(openmm.unit.kilojoule_per_mole)
                
                nonbonded_force.setParticleParameters(i, new_charge_dye_sg, sigma_val, epsilon_val)
                if verbose:    
                    print(f"  - New charge: {new_charge_dye_sg.value_in_unit(openmm.unit.elementary_charge):+.6f} e")
                found_sg = True

            elif atom.name == 'Fe1x':
                charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
                if verbose:    
                    print(f"Found DYE-Fe1x: Atom ID (Index) {i}")
                    print(f"  - Old charge: {charge.value_in_unit(openmm.unit.elementary_charge):+.6f} e")
                

                sigma_val = sigma.value_in_unit(openmm.unit.nanometer)
                epsilon_val = epsilon.value_in_unit(openmm.unit.kilojoule_per_mole)
                nonbonded_force.setParticleParameters(i, new_charge_dye_fe, sigma_val, epsilon_val)
                if verbose:
                    print(f"  - New charge: {new_charge_dye_fe.value_in_unit(openmm.unit.elementary_charge):+.6f} e")
                found_fe = True

    if verbose:    
        print("\nApplying changes to the simulation context...")
    nonbonded_force.updateParametersInContext(simulation.context)
    if verbose:
        print("Charges successfully updated in the simulation.")

    if not found_sg:
        print("Warning: Did not find atom 'SG' in residue 'DYE'.")
    if not found_fe:
        print("Warning: Did not find atom 'Fe1x' in residue 'DYE'.")
    traj = md.load(packed_pdb_path)
    topology = traj.topology
    
    fe_atom = topology.select('element Fe')
    


    N1_atom = topology.select("resname HEM and name N1x")
    N2_atom = topology.select("resname HEM and name N2x")
    N3_atom = topology.select("resname HEM and name N3x")
    N4_atom = topology.select("resname HEM and name N4x")

    N_Fe_pairs = [
        [[N1_atom[0], fe_atom[0]]],
        [[N2_atom[0], fe_atom[0]]],
        [[N3_atom[0], fe_atom[0]]],
        [[N4_atom[0], fe_atom[0]]],
    ]
    N_Fe_distances = [md.compute_distances(traj, p)[0] * openmm_unit.nanometer for p in N_Fe_pairs]

    # Angles
    N1_Fe_N4_triplet = [[N1_atom[0], fe_atom[0], N4_atom[0]]]
    N2_Fe_N3_triplet = [[N2_atom[0], fe_atom[0], N3_atom[0]]]
    N1_Fe_N4_angle = md.compute_angles(traj, N1_Fe_N4_triplet)[0] * openmm_unit.radians
    N2_Fe_N3_angle = md.compute_angles(traj, N2_Fe_N3_triplet)[0] * openmm_unit.radians

    k_N_Fe = 95395.2 * openmm_unit.kilojoule_per_mole / openmm_unit.nanometer**2
    
    bond_force = mm.HarmonicBondForce()

    bond_force.addBond(int(N1_atom[0]), int(fe_atom[0]), N_Fe_distances[0], k_N_Fe)
    bond_force.addBond(int(N2_atom[0]), int(fe_atom[0]), N_Fe_distances[1], k_N_Fe)
    bond_force.addBond(int(N3_atom[0]), int(fe_atom[0]), N_Fe_distances[2], k_N_Fe)
    bond_force.addBond(int(N4_atom[0]), int(fe_atom[0]), N_Fe_distances[3], k_N_Fe)
    simulation.system.addForce(bond_force)

    # Add angle forces
    kangl = 1999.952 * openmm_unit.kilojoule/(openmm_unit.mole*openmm_unit.radian**2)
    angle_force = mm.HarmonicAngleForce()
    angle_force.addAngle(int(N1_atom[0]), int(fe_atom[0]), int(N4_atom[0]), N1_Fe_N4_angle, kangl)
    angle_force.addAngle(int(N2_atom[0]), int(fe_atom[0]), int(N3_atom[0]), N2_Fe_N3_angle, kangl)
    simulation.system.addForce(angle_force)
    
    if verbose:
        print("...custom Heme-Fe constraints added.")
    return simulation

# --- Stage 5: Run Simulation ---
def _run_simulation(
    simulation: "openmm.app.Simulation",
    simulation_dir: str,
    n_nanoseconds: int,
    temperature_k: float,         
    pressure_bar: float,          
    write_interval_steps: int,    
    log_interval_steps: int,      
    verbose: bool = False
):
    """
    Minimizes, heats, and runs the production simulation.
    """
    if verbose:
        print("Stage 5: Running simulation...")

    # --- Minimization with frozen core ---
    traj = md.load(f"{simulation_dir}/packed_complex.pdb")
    fe_atom = int(traj.top.select('element Fe')[0])
    N1_atom = int(traj.top.select("resname HEM and name N1x")[0])
    N2_atom = int(traj.top.select("resname HEM and name N2x")[0])
    N3_atom = int(traj.top.select("resname HEM and name N3x")[0])
    N4_atom = int(traj.top.select("resname HEM and name N4x")[0])
    
    atoms_to_freeze = [fe_atom, N1_atom, N2_atom, N3_atom, N4_atom]

    
    original_masses = [simulation.system.getParticleMass(i) for i in atoms_to_freeze]
    
    for atom_idx in atoms_to_freeze:
        simulation.system.setParticleMass(atom_idx, 0)    
    
    if verbose:
        print("  Running constrained minimization...")
    simulation.minimizeEnergy(tolerance=10)
    if verbose:
        print("  Minimization complete. Unfreezing atoms.")
        
    for atom_idx, mass in zip(atoms_to_freeze, original_masses):
        simulation.system.setParticleMass(atom_idx, mass)

    # Save minimized topology
    with open(f"{simulation_dir}/topology_complex.pdb", "w") as pdb_file:
        app.PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
            file=pdb_file,
            keepIds=False,
        )

    # --- Setup Reporters ---
    simulation.reporters.append(
        md.reporters.XTCReporter(file=str(f"{simulation_dir}/trajectory_complex.xtc"), reportInterval=write_interval_steps)
    )
    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout if verbose else f"{simulation_dir}/run.log",
            log_interval_steps,
            step=True, potentialEnergy=True, temperature=True, speed=True, separator="\t",
        )
    )
    simulation.reporters.append(
        app.StateDataReporter(
            f"{simulation_dir}/log_E_temp_vol.txt",
            200, step=True, potentialEnergy=True, temperature=True, volume=True,
        )
    )
    
    # --- Heating Ramp ---
    temp_kelvin = temperature_k * openmm_unit.kelvin
    pressure = pressure_bar * openmm_unit.bar
    simulation.system.addForce(MonteCarloBarostat(pressure, temp_kelvin))
    simulation.context.reinitialize(preserveState=True)

    if verbose:
        print(f"  Heating system to {temperature_k} K over 100k steps...")
    for i in range(10):
        # Heat up to the target temperature
        temperature = (i / 9.0) * temp_kelvin
        simulation.context.setParameter(MonteCarloBarostat.Temperature(), temperature)
        simulation.integrator.setTemperature(temperature)
        simulation.step(10000)
    if verbose:
        print("  Heating complete.")

    # --- Production Run ---
    timestep = simulation.integrator.getStepSize()
    total_duration = n_nanoseconds * openmm_unit.nanoseconds
    num_steps = int(total_duration / timestep)
    
    if verbose:
        print(f"Running production for {n_nanoseconds} ns ({num_steps} steps)...")
    
    start_time = time.time()
    simulation.step(num_steps)
    stop_time = time.time()

    wall_clock_time_days = (stop_time - start_time) / (24 * 3600)
    if wall_clock_time_days == 0:
        wall_clock_time_days = 1e-6 # Avoid division by zero
        
    total_ns_simulated = (num_steps * timestep).value_in_unit(openmm_unit.nanoseconds)
    simulation_speed_ns_per_day = total_ns_simulated / wall_clock_time_days

    if verbose:
        print(f"Simulation time: {stop_time - start_time:.2f} s; simulation speed: {simulation_speed_ns_per_day:.2f} ns/day")    

# --- NEW ---
def _cleanup_files(
    simulation_dir: str,
    verbose: bool = False
):
    """
    Removes intermediate files created during setup.
    """
    if verbose:
        print("Stage 6: Cleaning up intermediate files...")

    # List of all intermediate files from _prepare_inputs
    files_to_delete = [
        "complex.pdb",
        "unprotonated_ligand.pdb",       # The initial input ligand file
        "ligand_bonds_fixed.sdf",
        "plain_protein.pdb",
        "fixed_plain_protein.pdb",
        "heme_without_fe_unprotonated.pdb",
        "protonated_heme_without_fe.sdf",
        "prepared_heme.sdf",
        "Fe.pdb",
        "Fe.sdf",
        "protein_with_CYF.pdb", # Also remove the new PTM PDB
    ]

    for filename in files_to_delete:
        try:
            filepath = os.path.join(simulation_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                if verbose:
                    print(f"  Removed: {filename}")
        except Exception as e:
            print(f"Warning: Could not remove file {filename}. Error: {e}")
    
    if verbose:
        print("...cleanup complete.")
# --- END NEW ---
    
# --- The Main "Manager" Function (Public) ---
def run_heme_simulation(
    pdb_file_path: str,
    ligand_smiles: str,
    ligand_name: str,
    output_dir: str,
    n_nanoseconds: int = 5,
    temperature_k: float = 310.15,
    pressure_bar: float = 1.0,
    timestep_fs: float = 2.0,
    friction_coeff_ps: float = 1.0,
    write_interval_steps: int = 10000,
    log_interval_steps: int = 10000,
    cleanup: bool = True, 
    verbose: bool = False
):
    """
    Prepares and runs a simulation for a Heme-protein-ligand complex.

    This function serves as the main entry point for the cyclops pipeline.
    It automates the entire process:
    
    1. Prepares inputs (protein, heme, ligand) from a PDB and SMILES.
    2. Parameterizes each component using Open Force Field.
    3. Assembles and solvates the system with water and counter-ions.
    4. Adds custom Heme-Fe-N constraints (S-Fe bond is now in the FF).
    5. Runs energy minimization, heating, and production MD.
    6. Cleans up intermediate files (if `cleanup` is True).

    Parameters
    ----------
    pdb_file_path : str
        Path to the input PDB file.
    ligand_smiles : str
        SMILES string for the ligand to be parameterized.
    ligand_name : str
        The 3-letter residue name for the ligand.
    output_dir : str
        Path to the directory where all simulation files will be saved.
    n_nanoseconds : int, optional
        The total length of the production simulation in nanoseconds.
        Default is 5.
    temperature_k : float, optional
        Target temperature for the simulation in Kelvin. Default is 310.15.
    pressure_bar : float, optional
        Target pressure for the simulation in bar. Default is 1.0.
    timestep_fs : float, optional
        The integration timestep in femtoseconds. Default is 2.0.
    friction_coeff_ps : float, optional
        The friction coefficient for the Langevin integrator in 1/picoseconds.
        Default is 1.0.
    write_interval_steps : int, optional
        How often (in steps) to save a frame to the XTC trajectory file.
        Default is 10000.
    log_interval_steps : int, optional
        How often (in steps) to write to the console/log file.
        Default is 10000.
    cleanup : bool, optional
        If True (default), all intermediate setup files (e.g.,
        "plain_protein.pdb", "prepared_ligand.sdf") will be
        deleted after the simulation finishes.
    verbose : bool, optional
        If True, prints detailed status messages to the console.
        Default is False.

    Returns
    -------
    None
        This function does not return any objects.

    Notes
    -----
    This function writes the following key files to the `output_dir`:
    
    - `packed_complex.pdb`: The solvated (t=0) system for analysis.
    - `topology_complex.pdb`: The minimized system topology.
    - `trajectory_complex.xtc`: The simulation trajectory.
    - `log_E_temp_vol.txt`: Log file for analysis.
    - `run.log`: Verbose log from OpenMM (if `verbose=False`).
    """
    start = time.time()
    
    # 1. Setup directories
    simulation_dir = output_dir
    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)
        
    # 2. Prepare inputs
    paths = _prepare_inputs(pdb_file_path, ligand_smiles, ligand_name, simulation_dir, verbose)
    
    # 3. Parameterize
    components = _parameterize_components(paths, ligand_name, verbose)

    # 4. Assemble and Solvate
    final_interchange = _assemble_and_solvate(components, simulation_dir, verbose)
    
    # 5. Create Simulation Object
    temp_with_unit = temperature_k * openmm_unit.kelvin
    friction_with_unit = friction_coeff_ps / openmm_unit.picosecond
    timestep_with_unit = timestep_fs * openmm_unit.femtoseconds
    
    integrator = openmm.LangevinIntegrator(
        temp_with_unit,
        friction_with_unit,
        timestep_with_unit,
    )
    simulation = final_interchange.to_openmm_simulation(integrator=integrator)

    # 6. Add Custom Constraints
    _add_custom_constraints(simulation, f"{simulation_dir}/packed_complex.pdb", verbose)

    # 7. Run
    _run_simulation(
        simulation, 
        simulation_dir, 
        n_nanoseconds,
        temperature_k,
        pressure_bar,
        write_interval_steps,
        log_interval_steps,
        verbose
    )
    

    if cleanup:
        _cleanup_files(simulation_dir, verbose)

    
    
    stop = time.time()
    if verbose:
        print(f'\n--- Total execution time: {stop-start:.2f} seconds ---')