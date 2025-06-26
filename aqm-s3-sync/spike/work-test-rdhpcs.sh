#!/bin/bash

set -xue

source ../../../gaea/env.sh
export PATH=/gpfs/f6/bil-fire8/scratch/Benjamin.Koziol/sandbox/miniconda3/envs/benkozi-work/bin:${PATH}

rm -rf /ncrc/home2/Benjamin.Koziol/l/scratch/tmp/aqm-use-case-download/*
git pull
${BWK_CONDA_RUN} pytest -s test_aqm_data_sync.py::test_main_gaeac6