from pathlib import Path

import typer

from aqm_eval.aqm_mm_eval.driver.interface import SRWInterface


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
) -> None:
    print("in exsrw_aqm_mm.py")
    print(f"{expt_dir=}")
    srw_interface = SRWInterface(expt_dir=expt_dir)
    print(f"{srw_interface=}")
    raise ValueError("done")


if __name__ == '__main__':
    typer.run(main)