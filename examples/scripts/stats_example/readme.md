## Statistical Analysis

This statistical summary script analyzes the trajectory data. **Note:** Before running this analysis, ensure you have first generated the necessary input files using the extraction scripts located in the `../scripts` directory.

### Environment Setup

To run the analysis cleanly, create an isolated Conda environment with all the required dependencies. You can set this up in three quick steps:

**1. Create the environment and install core packages:**
```bash
conda create -n md_analysis -c conda-forge python=3.10 pandas numpy matplotlib seaborn scipy pingouin statsmodels -y
```

**2. Activate the new environment:**
```bash
conda activate md_analysis
```

**3. Install the remaining plotting libraries:**
```bash
pip install scienceplots
```