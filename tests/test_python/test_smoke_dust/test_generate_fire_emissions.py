import logging
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

logger = logging.getLogger('test_generate_fire_emissions')
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(logging.Formatter('[%(levelname)s][%(name)s][%(created)f] %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)  # tdk:set to info


@dataclass
class GenerateEmissWorkflowArgs:
    staticdir: str
    ravedir: str
    intp_dir: str
    predef_grid: str
    ebb_dcycle_flag: str
    restart_interval: str
    persistence: str

    @classmethod
    def create(cls, basedir: Path) -> "GenerateEmissWorkflowArgs":
        return cls(staticdir=str(basedir / 'staticdir'),
                   ravedir=str(basedir / 'ravedir'),
                   intp_dir=str(basedir / 'intpdir'),
                   predef_grid='RRFS_NA_3km',  # tdk: test with other grid
                   ebb_dcycle_flag='1',  # tdk: test with 2
                   restart_interval='6 12 18 24',
                   persistence='TRUE'  # tdk: test with false
                   )

    def as_script_args(self) -> Tuple:
        return self.staticdir, self.ravedir, self.intp_dir, self.predef_grid, self.ebb_dcycle_flag, self.restart_interval, self.persistence


class TestGenerateFireEmissions(unittest.TestCase):

    def __init__(self, *args, **kwargs) -> None:
        self._ushdir = Path("../../../ush")
        assert self._ushdir.name == "ush"
        super().__init__(*args, **kwargs)

    def setUp(self) -> None:
        self._temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self._temp_dir)

    def test(self) -> None:
        main_args = GenerateEmissWorkflowArgs.create(self._temp_dir)
        logger.debug(main_args)
        main_path = self._ushdir / "generate_fire_emissions.py"
        subprocess.check_call(['python3', main_path] + list(main_args.as_script_args()))
