#!/usr/bin/env bash

set -e

# Comment to skip regridding (if it has been completed)
#rm ~/htmp/comout/intp_dir/* || echo "no interpolation data to remove"

cd ~/l/test_smoke_dust-python
git pull
rm *.err *.out || echo "no job logs to remove"

#bash ./job-run-test_generate_fire_emissions.sh

sbatch job-run-test_generate_fire_emissions.sh
squeue -u Benjamin.Koziol -i 5
