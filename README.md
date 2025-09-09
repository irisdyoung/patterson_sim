## Patterson map simulator

Starting from a PDB model, generate Cartesian coordinates at the positions of the expected Patterson peaks.

### Setup

To construct the necessary conda environment, do the following:

```
conda create -n patterson_sim
conda activate patterson_sim
conda install -c conda-forge gemmi python=3.13 mpi4py openmpi
```

If you will use MPI, ensure you have that set up correctly for your machine. On Ubuntu, run `sudo apt install python3-mpi4py`; see https://mpi4py.readthedocs.io/en/stable/install.html for installation on other systems.


