#!/usr/bin/env python3

"""
Python script for fire emissions preprocessing from RAVE FRP and FRE (Li et al.,2022)
Author: johana.romero-alvarez@noaa.gov
"""

import sys
from pathlib import Path

import typer

sys.path.append(str(Path(__file__).parent.parent))

from smoke_dust.core.context import PredefinedGrid, EbbDCycle, RaveQaFilter, LogLevel, \
    SmokeDustContext
from smoke_dust.core.preprocessor import SmokeDustPreprocessor


def main(
    staticdir: Path = typer.Option(
        ..., "--staticdir", help="Path to the smoke and dust fixed files."
    ),
    ravedir: Path = typer.Option(
        ..., "--ravedir", help="Path to the directory containing RAVE data files (hourly)."
    ),
    intp_dir: Path = typer.Option(
        ..., "--intp-dir", help="Path to the directory containing interpolated RAVE data files."
    ),
    predef_grid: PredefinedGrid = typer.Option(
        ..., "--predef-grid", help="SRW predefined grid to use as the forecast domain."
    ),
    ebb_dcycle: EbbDCycle = typer.Option(..., "--ebb-dcycle", help="The forecast cycle to run."),
    restart_interval: str = typer.Option(
        ...,
        "--restart-interval",
        help="Restart intervals used for restart file search. For example '6 12 18 24'.",
    ),
    persistence: bool = typer.Option(
        ...,
        "--persistence",
        help="If true, use satellite observations from the previous day. Otherwise, use observations from the same day.",
    ),
    rave_qa_filter: RaveQaFilter = typer.Option(
        ..., "--rave-qa-filter", help="Filter level for RAVE QA flags when regridding fields."
    ),
    log_level: LogLevel = typer.Option(
        LogLevel.INFO, "--log-level", help="Logging level to use for the preprocessor."
    ),
    exit_on_error: bool = typer.Option(
        True,
        "--exit-on-error",
        help="If false, log errors and write a dummy emissions file but do not raise an exception.",
    ),
    regrid_in_memory: bool = typer.Option(
        False,
        "--regrid-in-memory",
        help="If true, do esmpy regridding in-memory as opposed to reading from the fixed weight file.",
    ),
):
    typer.echo("Welcome to interpolating RAVE and processing fire emissions!")

    context = SmokeDustContext(
        staticdir=staticdir,
        ravedir=ravedir,
        intp_dir=intp_dir,
        predef_grid=predef_grid,
        ebb_dcycle=ebb_dcycle,
        restart_interval=restart_interval,
        persistence=persistence,
        rave_qa_filter=rave_qa_filter,
        log_level=log_level,
        exit_on_error=exit_on_error,
        regrid_in_memory=regrid_in_memory,
    )
    processor = SmokeDustPreprocessor(context)
    try:
        processor.run()
        processor.finalize()
    except Exception as e:  # pylint: disable=broad-exception-caught
        processor.create_dummy_emissions_file()
        context.log("unhandled error", exc_info=e)

    typer.echo("Exiting. Bye!")


if __name__ == "__main__":
    typer.run(main)
