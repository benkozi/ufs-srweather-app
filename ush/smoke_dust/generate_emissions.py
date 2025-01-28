#!/usr/bin/env python3

#########################################################################
#                                                                       #
# Python script for fire emissions preprocessing from RAVE FRP and FRE  #
# (Li et al.,2022).                                                     #
# johana.romero-alvarez@noaa.gov                                        #
#                                                                       #
#########################################################################

import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent.parent))

from smoke_dust.core.context import SmokeDustContext
from smoke_dust.core.preprocessor import SmokeDustPreprocessor


def main(args: List[str]) -> None:
    """
    Prepares fire-related ICs. This is the main function that handles data movement and interpolation.
    #tdk: doc
    Args:
        staticdir: Path to fix files for the smoke and dust component
        ravedir: Path to the directory containing RAVE fire data files (hourly). This is typically the working directory (DATA)
        intp_dir: Path to interpolated RAVE data files from the previous cycles (DATA_SHARE)
        predef_grid: If ``RRFS_NA_3km``, use pre-defined grid dimensions
        ebb_dcycle_flag: Select the EBB cycle to run. Valid values are ``"1"`` or ``"2"``
        restart_interval: Indicates if restart files should be copied. The actual interval values are not used
        persistence: If ``TRUE``, use satellite observations from the previous day. Otherwise, use observations from the same day.
    """

    context = SmokeDustContext.create_from_args(args)
    processor = SmokeDustPreprocessor(context)
    try:
        processor.run()
        processor.finalize()
    except Exception as e:
        processor.create_dummy_emissions_file()
        context.log("unhandled error", exc_info=e)


if __name__ == "__main__":
    print("")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Welcome to interpolating RAVE and processing fire emissions!")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")
    # tdk:story: use argparse
    main(sys.argv[1:])
    print("")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Exiting. Bye!")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")
