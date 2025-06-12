#!/usr/bin/env bash

#
#-----------------------------------------------------------------------
#
# Source the variable definitions file and the bash utility functions.
#
#-----------------------------------------------------------------------
#
. ${USHsrw}/source_util_funcs.sh
sections=(
  user
  nco
  platform
  workflow
  global
  verification
  cpl_aqm_parm
  constants
  fixed_files
  grid_params
)
for sect in ${sections[*]} ; do
  source_yaml ${GLOBAL_VAR_DEFNS_FP} ${sect}
done
#
#-----------------------------------------------------------------------
#
# Save current shell options (in a global array).  Then set new options
# for this script/function.
#
#-----------------------------------------------------------------------
#
{ save_shell_opts; set -xue; } > /dev/null 2>&1
#
#-----------------------------------------------------------------------
#
# Get the full path to the file in which this script/function is located 
# (scrfunc_fp), the name of that file (scrfunc_fn), and the directory in
# which the file is located (scrfunc_dir).
#
#-----------------------------------------------------------------------
#
scrfunc_fp=$( $READLINK -f "${BASH_SOURCE[0]}" )
scrfunc_fn=$( basename "${scrfunc_fp}" )
scrfunc_dir=$( dirname "${scrfunc_fp}" )
#
#-----------------------------------------------------------------------
#
# Print message indicating entry into script.
#
#-----------------------------------------------------------------------
#
print_info_msg "
========================================================================
Entering script:  \"${scrfunc_fn}\"
In directory:     \"${scrfunc_dir}\"

This is the ex-script for the task that fetches fire emission
data files from disk or generates model-ready RAVE emission file from raw
data files.
========================================================================"
#
#-----------------------------------------------------------------------
#
# Set up variables for call to retrieve_data.py
#
#-----------------------------------------------------------------------
#
YYYYMMDD=${FIRE_FILE_CDATE:0:8}
HH=${FIRE_FILE_CDATE:8:2}

CDATE_mh1=`$NDATE -1 ${YYYYMMDD}${HH}`
yyyymmdd_mh1=${CDATE_mh1:0:8}
hh_mh1=${CDATE_mh1:8:2}
#
#-----------------------------------------------------------------------
#
# Retrieve fire file to FIRE_EMISSION_STAGING_DIR
#
#-----------------------------------------------------------------------
#

#tdk:aqm-data
if [ "${USE_AQM_S3_DATA_STAGE}" = "True" ]; then
  echo "tdk: retrieving data from s3 cloud storage"
  yyyymmdd=${PDY}
  hh=${cyc}
  EXTRN_MDL_CDATE=$( $DATE_UTIL --utc --date "${yyyymmdd} ${hh} UTC - ${TIME_OFFSET_HRS} hours" "+%Y%m%d%H" )
  EXTRN_MDL_NAME="UFS-AQM-FV3GFS"
  EXTRN_MDL_STAGING_DIR="${COMIN}/${EXTRN_MDL_NAME}/"
  mkdir -p "${EXTRN_MDL_STAGING_DIR}"
  cmd="
  python3 -u ${USHdir}/retrieve_data.py \
    --debug \
    --file_set anl \
    --config ${PARMdir}/data_locations.yml \
    --cycle_date ${EXTRN_MDL_CDATE} \
    --data_stores aws \
    --data_type ${EXTRN_MDL_NAME} \
    --fcst_hrs ${TIME_OFFSET_HRS} \
    --output_path ${EXTRN_MDL_STAGING_DIR} \
    --summary_file "${EXTRN_MDL_VAR_DEFNS_FN}.sh" \
    --ics_or_lbcs ICS \
    --file_fmt netcdf \
    "
  $cmd
  export err=$?
  if [ $err -ne 0 ]; then
    message_txt="Call to retrieve_data.py failed with a non-zero exit status.
  The command was:
  ${cmd}
  "
      print_err_msg_exit "${message_txt}"
  fi
  exit 1
fi

aqm_fire_file_fn="${AQM_FIRE_FILE_PREFIX}_${YYYYMMDD}_t${HH}z${AQM_FIRE_FILE_SUFFIX}"

# Check if the fire file exists in the designated directory
if [ -e "${COMINfire}/${aqm_fire_file_fn}" ]; then
  cp -p "${COMINfire}/${aqm_fire_file_fn}" ${COMOUT}
else
  # Copy raw data 
  for ihr in {0..23}; do
    download_time=`$NDATE -$ihr ${yyyymmdd_mh1}${hh_mh1}`
    FILE_curr="Hourly_Emissions_13km_${download_time}00_${download_time}00.nc"
    FILE_13km="RAVE-HrlyEmiss-13km_v*_blend_s${download_time}00000_e${download_time}59590_c*.nc"
    yyyymmdd_dn="${download_time:0:8}"
    hh_dn="${download_time:8:2}"
    missing_download_time=`$NDATE -24 ${yyyymmdd_dn}${hh_dn}`
    yyyymmdd_dn_md1="${missing_download_time:0:8}"
    FILE_13km_md1="RAVE-HrlyEmiss-13km_v*_blend_s${missing_download_time}00000_e${missing_download_time}59590_c*.nc"
    if [ -s `ls ${COMINfire}/${yyyymmdd_dn}/rave/${FILE_13km}` ] && [ $(stat -c %s `ls ${COMINfire}/${yyyymmdd_dn}/rave/${FILE_13km}`) -gt 4000000 ]; then
      cp -p ${COMINfire}/${yyyymmdd_dn}/rave/${FILE_13km} ${FILE_curr}
    elif [ -s `ls ${COMINfire}/${yyyymmdd_dn_md1}/rave/${FILE_13km_md1}` ] && [ $(stat -c %s `ls ${COMINfire}/${yyyymmdd_dn_md1}/rave/${FILE_13km_md1}`) -gt 4000000 ]; then
      echo "WARNING: ${FILE_13km} does not exist or broken. Replacing with the file of previous date ..."
      cp -p ${COMINfire}/${yyyymmdd_dn_md1}/rave/${FILE_13km_md1} ${FILE_curr}
    else
      message_txt="WARNING Fire Emission RAW data does not exist or broken:
  FILE_13km_md1 = \"${FILE_13km_md1}\"
  COMINfire = \"${COMINfire}\""

      cp -p ${FIXaqm}/fire/Hourly_Emissions_13km_dummy.nc ${FILE_curr}
      print_info_msg "WARNING: ${message_txt}. Replacing with the dummy file :: AQM RUN SOFT FAILED."
    fi
  done  

  ncks -O -h --mk_rec_dmn time Hourly_Emissions_13km_${download_time}00_${download_time}00.nc temp.nc
  export err=$?
  if [ $err -ne 0 ]; then
    message_txt="Call to NCKS returned with nonzero exit code."
    err_exit "${message_txt}"
    print_err_msg_exit "${message_txt}"
  fi

  mv temp.nc Hourly_Emissions_13km_${download_time}00_${download_time}00.nc

  ncrcat -h Hourly_Emissions_13km_*.nc Hourly_Emissions_13km_${YYYYMMDD}0000_${YYYYMMDD}2300.t${HH}z.nc
  export err=$?
  if [ $err -ne 0 ]; then
    message_txt="Call to NCRCAT returned with nonzero exit code."
    err_exit "${message_txt}"
    print_err_msg_exit "${message_txt}"
  fi

  input_fire="${DATA}/Hourly_Emissions_13km_${YYYYMMDD}0000_${YYYYMMDD}2300.t${HH}z.nc"
  output_fire="${DATA}/Hourly_Emissions_regrid_NA_13km_${YYYYMMDD}_new24.t${HH}z.nc"

  ${USHsrw}/aqm_utils_python/RAVE_remake.allspecies.aqmna13km.g793.py --date "${YYYYMMDD}" --cyc "${HH}" --input_fire "${input_fire}" --output_fire "${output_fire}"
  export err=$?
  if [ $err -ne 0 ]; then
    message_txt="Call to python script \"RAVE_remake.allspecies.py\" returned with nonzero exit code."
    err_exit "${message_txt}"
    print_err_msg_exit "${message_txt}"
  fi

  ncks --mk_rec_dmn Time Hourly_Emissions_regrid_NA_13km_${YYYYMMDD}_new24.t${HH}z.nc -o Hourly_Emissions_regrid_NA_13km_${YYYYMMDD}_t${HH}z_h24.nc
  export err=$?
  if [ $err -ne 0 ]; then
    message_txt="Call to NCKS returned with nonzero exit code."
    err_exit "${message_txt}"
    print_err_msg_exit "${message_txt}"
  fi

  cp -p Hourly_Emissions_regrid_NA_13km_${YYYYMMDD}_t${HH}z_h24.nc Hourly_Emissions_regrid_NA_13km_${YYYYMMDD}_t${HH}z_h24_1.nc
  cp -p Hourly_Emissions_regrid_NA_13km_${YYYYMMDD}_t${HH}z_h24.nc Hourly_Emissions_regrid_NA_13km_${YYYYMMDD}_t${HH}z_h24_2.nc

  ncrcat -O -D 2 Hourly_Emissions_regrid_NA_13km_${YYYYMMDD}_t${HH}z_h24.nc Hourly_Emissions_regrid_NA_13km_${YYYYMMDD}_t${HH}z_h24_1.nc Hourly_Emissions_regrid_NA_13km_${YYYYMMDD}_t${HH}z_h24_2.nc ${aqm_fire_file_fn}
  export err=$?
  if [ $err -ne 0 ]; then
    message_txt="Call to NCRCAT returned with nonzero exit code."
    err_exit "${message_txt}"
    print_err_msg_exit "${message_txt}"
  fi

  mv ${aqm_fire_file_fn}  temp.nc
  ncrename -v PM2.5,PM25 temp.nc temp1.nc
  ncap2 -s 'where(Latitude > 30 && Latitude <=49 && land_cover == 1 ) PM25 = PM25 * 0.44444' temp1.nc temp2.nc
  ncap2 -s 'where(Latitude <=30 && land_cover == 1 ) PM25 = PM25 * 0.8'       temp2.nc temp3.nc
  ncap2 -s 'where(Latitude <=49 && land_cover == 3 ) PM25 = PM25 * 1.11111'   temp3.nc temp4.nc
  ncap2 -s 'where(Latitude <=49 && land_cover == 4 ) PM25 = PM25 * 1.11111'   temp4.nc temp5.nc
  ncrename -v PM25,PM2.5 temp5.nc temp6.nc
  mv temp6.nc ${aqm_fire_file_fn}

  # Copy the final fire emission file to data share directory
  cp -p "${DATA}/${aqm_fire_file_fn}" ${COMOUT}
fi
#
#-----------------------------------------------------------------------
#
# Restore the shell options saved at the beginning of this script/function.
#
#-----------------------------------------------------------------------
#
{ restore_shell_opts; } > /dev/null 2>&1
