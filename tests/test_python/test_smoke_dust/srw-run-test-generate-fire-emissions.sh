#!/usr/bin/env bash
#
#SBATCH --job-name=job-test_generate_fire_emissions
#SBATCH --account=epic
#SBATCH --qos=batch
#SBATCH --partition=hera
#_SBATCH --partition=bigmem
#SBATCH -t 00:10:00
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#_SBATCH --output=%x_%j.out
#_SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8  # Assuming 24 cores per node, utilize them fully
#SBATCH --ntasks=8  # Total tasks should be nodes * tasks-per-node

set -e

# orion
#TESTDIR=/work/noaa/epic/bwkoziol/sandbox/srw-main-aqm/predefined-grids/ufs-srweather-app/tests/test_python/test_smoke_dust
#CONDA_ENV=/work/noaa/epic/bwkoziol/sandbox/miniconda3/envs/regrid-wrapper

# hera
TESTDIR=/scratch2/NAGAPE/epic/Ben.Koziol/sandbox/srw-main-aqm/predefined-grids/ufs-srweather-app/tests/test_python/test_smoke_dust
CONDA_ENV=/scratch2/NAGAPE/epic/Ben.Koziol/miniconda/envs/regrid-wrapper

# ----------

export ESMFMKFILE=${CONDA_ENV}/lib/esmf.mk
export PATH=${CONDA_ENV}/bin:${PATH}

# Comment to skip regridding
rm ~/htmp/comout/intp_dir/* || echo "no interpolation data to remove"

cd ${TESTDIR}
#git pull

echo "running python unit test"
#python -m unittest ${TESTDIR}/test_generate_fire_emissions.py
mpirun -n 24 python -m unittest ${TESTDIR}/test_generate_fire_emissions.py
