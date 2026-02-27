Getting Started
===============

Installation
------------

The recommended installation method is using **Conda**.

1. **Clone the repository:**

   .. code-block:: bash

      git clone git@github.com:hruska-lab/package_cyclops_dev.git
      cd package_cyclops_dev/

2. **Create the Conda Environment:**
   This command uses the provided environment file to create a new environment named ``cyclops-dev`` with all the correct dependencies.

   .. code-block:: bash

      conda env create -f environment-dev.yml

3. **Activate the Environment:**

   .. code-block:: bash

      conda activate cyclops-dev

4. **Install cyclops:**

   .. code-block:: bash

      pip install -e .

   *Note: In case of dependency conflict coming from openbabel:*

   .. code-block:: bash

      pip install -e . --no-deps

Quick Start: Running a Simulation
---------------------------------

You can run a simulation by importing and using the ``run_heme_simulation`` function from the ``cyclops`` package.

.. code-block:: python

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