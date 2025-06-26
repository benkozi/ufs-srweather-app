import subprocess
from pathlib import Path

from typer.testing import CliRunner

from ads.ads_cli import app
from ads.core import UseCaseKey


def test_help() -> None:
    """Test that the help message can be displayed."""
    cli_path = Path(__file__).parent.parent / "ads" / "ads_cli.py"
    subprocess.check_call(["python", str(cli_path), "--help"])


def test_use_case(tmp_path: Path) -> None:
    runner = CliRunner()

    args = [
        "--use-case",
        UseCaseKey.AEROMMA.value,
        "--dst-dir",
        str(tmp_path),
        "--dry-run",
    ]
    result = runner.invoke(app, args, catch_exceptions=False)
    print(result.output)
