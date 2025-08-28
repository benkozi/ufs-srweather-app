import os
from pathlib import Path

import typer


def main(
    expt_dir: Path = typer.Option(
        ...,
        "--expt-dir", 
        help="Path to experiment directory",
        exists=True,
        readable=True,
        file_okay=False,
        dir_okay=True
    )
):

    raise ValueError('in exsrw_aqm_mm.py')


if __name__ == '__main__':
    typer.run(main)