This repository includes several tools to help you get started with **Cyclops**:

### Interactive Examples

- **`simulations.ipynb`**
  A Jupyter notebook demonstrating a simple workflow. Use this to understand the basics of the API.
  - *Dependencies:* This notebook utilizes files stored in `examples_structures/`.

### Automation Scripts (`scripts/`)

We provide production-ready scripts to automate your workflow:

1. **Molecular Dynamics (`cyclops_md.py`)**
   A simulation-ready wrapper that reads structure information from a CSV file. This is ideal for batch-processing multiple structures in a single run. 


2. **Parallel Analysis (`cyclops_analysis_parallel.py`)**
   A performance-optimized wrapper for the `cyclops.analysis` module. It leverages Python's `concurrent` module to analyze multiple simulation results in parallel, significantly reducing processing time.

Both files can use a csv similar to `simulation_data.csv`, for which the input files are hosted separately FIND A PLACE FOR THEM AND COME BACK WITH A LINK
