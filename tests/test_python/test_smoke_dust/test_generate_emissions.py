import os
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from smoke_dust.core.context import RaveQaFilter
from smoke_dust.generate_emissions import app
from test_python.test_smoke_dust.conftest import create_fake_grid_out, FakeGridOutShape


def test(tmp_path: Path, fake_grid_out_shape: FakeGridOutShape) -> None:
    create_fake_grid_out(tmp_path, fake_grid_out_shape)
    strpath = str(tmp_path)
    runner = CliRunner()
    os.environ["CDATE"] = "2019072200"
    os.environ["DATA"] = strpath

    try:
        args = ["--staticdir", strpath, "--ravedir", strpath, "--intp-dir", strpath,
                 "--predef-grid", "RRFS_CONUS_25km", "--ebb-dcycle", "2",
                 "--restart-interval", "6 12 18 24", "--persistence", "false", "--rave-qa-filter",
                 "NONE"]
        print(args)
        result = runner.invoke(app,
                               args)
    except:
        for ii in ["CDATE", "DATA"]:
            os.unsetenv(ii)
        raise
    print(result.output)
    # print(result.exc_info[1])

    assert result.exit_code == 0


def test2(tmp_path: Path):
    strpath = str(tmp_path)
    os.environ["CDATE"] = "2019072200"
    os.environ["DATA"] = strpath
    args = ["--staticdir", strpath, "--ravedir", strpath, "--intp-dir", strpath,
            "--predef-grid", "RRFS_CONUS_25km", "--ebb-dcycle", "2",
            "--restart-interval", "6 12 18 24", "--persistence", "false", "--rave-qa-filter",
            "NONE"]
    subprocess.check_call(["python", "../../../ush/smoke_dust/generate_emissions.py", ] + args)