CYCLOPS
==============================
[//]: # (Badges)
<!--
([![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/cyclops/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/cyclops/actions?query=workflow%3ACI))
([![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/CYCLOPS/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/CYCLOPS/branch/main))
-->

CYtochrome Complex Ligand Optimization with Protein Simulation


## Installation

The recommended installation method is using **Conda**.

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:hruska-lab/CYCLOPS.git 
    cd CYCLOPS/
    ```

2.  **Create the Conda Environment:**
    This command uses the provided environment file to create a new environment named `cyclops` with all the correct dependencies.
    ```bash
    conda env create -f env.yml
    ```

3.  **Activate the Environment:**
    You must activate this environment
    ```bash
    conda activate cyclops
    ```

4.  **Install `cyclops`:**
    ```bash
    pip install -e .
    ```
In case of dependency conflict coming from openbabel:

	pip install -e . --no-deps

## Quick Start: Running a Simulation

You can run a simulation by importing and using the `run_heme_simulation` function from the `cyclops` package.

    from cyclops.cyclops import run_heme_simulation
    import os

    # 1. Define your inputs
    # (Using the test file as an example)
    pdb_file = "cyclops/tests/1W0E-4NY4.pdb"
    ligand_name = "UNL"
    ligand_smiles = "Cc1cc(ccc1OC)[C@]2(C3=NCC(CN3C(=N2)N)(F)F)c4ccc(c(c4)c5cccnc5)F"
    output_dir = "my_simulation"
    
    # 2. Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. Run the simulation
    run_heme_simulation(
        pdb_file_path=pdb_file,
        ligand_smiles=ligand_smiles,
        ligand_name=ligand_name,
        output_dir=output_dir,
        n_nanoseconds=5,  # Run for 5 ns
        verbose=True
    )
    
Additional material with more advanced examples available in the documentation.
## How to Cite

If you use this software in your research, please cite:

> Grenda, P., Ogos, M., & Hruška, E. (2025). *Automated Structural Refinement of Docked Complexes in Cytochrome P450 Using Molecular Dynamics*. ChemRxiv. https://doi.org/10.26434/chemrxiv-2025-mvv4k-v3

<details>
<summary>BibTeX</summary>

```bibtex
@article{doi:10.26434/chemrxiv-2025-mvv4k-v3,
  author  = {Grenda, Przemys{\l}aw and Ogos, Martyna and Hru{\v{s}}ka, Eugen},
  title   = {Automated Structural Refinement of Docked Complexes in Cytochrome P450 Using Molecular Dynamics},
  journal = {ChemRxiv},
  year    = {2025},
  doi     = {10.26434/chemrxiv-2025-mvv4k-v3},
  url     = {https://chemrxiv.org/doi/abs/10.26434/chemrxiv-2025-mvv4k-v3}
}
```

</details>

#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.
