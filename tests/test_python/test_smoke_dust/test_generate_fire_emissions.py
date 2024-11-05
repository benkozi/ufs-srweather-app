import logging
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

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
    cdate: str
    data: str

    @classmethod
    def create(cls, basedir: Path) -> "GenerateEmissWorkflowArgs":
        return cls(staticdir=str(basedir / 'staticdir'),
                   ravedir=str(basedir / 'ravedir'),
                   intp_dir=str(basedir / 'intpdir'),
                   predef_grid='RRFS_NA_3km',  # tdk: test with other grid
                   ebb_dcycle_flag='1',  # tdk: test with 2
                   restart_interval='6 12 18 24',
                   persistence='TRUE',  # tdk: test with false
                   cdate='2019072200',
                   data=str(basedir / 'data')
                   )

    def as_script_args(self) -> Tuple:
        return self.staticdir, self.ravedir, self.intp_dir, self.predef_grid, self.ebb_dcycle_flag, self.restart_interval, self.persistence

    @contextmanager
    def run_context(self) -> Dict[str, str]:
        l = logger.getChild('run_context')
        dirs = [self.staticdir, self.ravedir, self.intp_dir, self.data]
        for ii in dirs:
            l.debug(f'creating directory: {ii}')
            os.mkdir(ii)
        env_vars = {'CDATE': self.cdate, 'DATA': self.data}
        l.debug(f'setting environment variables: {env_vars}')
        orig = {k: os.environ.get(k) for k in env_vars.keys()}
        try:
            os.environ['CDATE'] = self.cdate
            os.environ['DATA'] = self.data
            yield env_vars
        finally:
            for k, v in orig.items():
                if v is not None:
                    os.environ[k] = v


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
        with main_args.run_context() as _:
            subprocess.check_call(['python3', main_path] + list(main_args.as_script_args()))
