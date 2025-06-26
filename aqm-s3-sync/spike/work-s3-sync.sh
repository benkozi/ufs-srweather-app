#!/bin/bash

aws configure set default.s3.max_concurrent_requests 3

aws s3 sync --no-sign-request s3://noaa-ufs-srw-pds/UFS-AQM/RAVE_fire .

aws s3 sync --dryrun --no-sign-request --exclude "*" --include "*20230601*" --include "*RESTART*20230531*" s3://noaa-ufs-srw-pds/UFS-AQM .