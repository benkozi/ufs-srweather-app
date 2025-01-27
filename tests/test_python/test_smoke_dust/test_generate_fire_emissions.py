import logging
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Literal

from smoke_dust.main import main

logger = logging.getLogger("test_generate_fire_emissions")
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(
    logging.Formatter("[%(levelname)s][%(name)s][%(created)f] %(message)s")
)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)  # tdk:set to info


@dataclass
class GenerateEmissWorkflowArgs:
    staticdir: Path
    ravedir: Path
    intp_dir: Path
    predef_grid: str
    ebb_dcycle_flag: str
    restart_interval: str
    persistence: str
    exit_on_error: str
    cdate: str
    data: Path
    rave_qa_filter: Literal["NONE", "HIGH"]
    log_level: str = "DEBUG"

    @classmethod
    def create(cls, comin: Path, comout: Path) -> "GenerateEmissWorkflowArgs":
        return cls(
            # staticdir=comin / 'RRFS_NA_13km',  # tdk: test with other grids
            # staticdir=comin / 'RRFS_CONUS_25km',  # tdk: test with other grids
            # staticdir=comin / 'RRFS_CONUS_13km', #tdk: test with other grids
            staticdir=comin / "RRFS_CONUS_3km",  # tdk: test with other grids
            # ravedir=comin / 'RAVE_fire',
            ravedir=Path(
                "/scratch2/NAGAPE/epic/SRW-AQM_DATA/data_smoke_dust/RAVE_fire"
            ),  # hera
            # ravedir=Path('/work/noaa/epic/SRW-AQM_DATA/data_smoke_dust/RAVE_fire'), #orion
            # tdk: make this configurable
            intp_dir=comout / "intp_dir",
            # predef_grid='RRFS_NA_13km',  # tdk: test with all grids
            # predef_grid='RRFS_CONUS_25km',  # tdk: test with all grids
            # predef_grid='RRFS_CONUS_13km',  # tdk: test with all grids
            predef_grid="RRFS_CONUS_3km",  # tdk: test with all grids
            # ebb_dcycle_flag='1',  # tdk: test with 2
            ebb_dcycle_flag="2",  # tdk: test with 2
            restart_interval="6 12 18 24",
            persistence="FALSE",  # tdk: test with false
            # persistence='TRUE',  # tdk: test with false
            cdate="2019072200",
            # data=comout / 'data',
            data="/scratch2/NAGAPE/epic/Ben.Koziol/sandbox/srw-main-aqm/control/nco_dirs/test_smoke/tmp/forecast_mem000.2019072206.5556738",  # tdk: need to figure out data directory
            exit_on_error="TRUE",  # tdk: test with FALSE
            rave_qa_filter="NONE",
        )

    def as_script_args(self) -> Tuple:
        return (
            str(self.staticdir),
            str(self.ravedir),
            str(self.intp_dir),
            self.predef_grid,
            self.ebb_dcycle_flag,
            self.restart_interval,
            self.persistence,
            self.rave_qa_filter,
            self.exit_on_error,
            self.log_level,
        )

    @contextmanager
    def run_context(self) -> Dict[str, str]:
        l = logger.getChild("run_context")

        # dirs = [self.intp_dir, self.data, self.staticdir, self.ravedir]
        # for ii in dirs:
        #     l.debug(f'creating directory: {ii}')
        #     os.mkdir(ii)

        env_vars = {"CDATE": self.cdate, "DATA": self.data}
        l.debug(f"setting environment variables: {env_vars}")
        orig = {k: os.environ.get(k) for k in env_vars.keys()}
        try:
            os.environ["CDATE"] = self.cdate
            os.environ["DATA"] = str(self.data)
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
        comin = Path("/scratch2/NAGAPE/epic/SRW-AQM_DATA/fix_smoke")  # hera
        # comin = Path("/home/bwkoziol/tmp-smoke-dust-fixed-files/") # orion
        # comin = Path(self._temp_dir)

        # comout = "/home/Benjamin.Koziol/htmp/comout" #hera
        # comout = "/home/bwkoziol/htmp/comout" #orion
        comout = "~/htmp/comout"

        main_args = GenerateEmissWorkflowArgs.create(comin, Path(comout))
        logger.debug(main_args)
        main_path = self._ushdir / "smoke_dust_main.py"
        with main_args.run_context() as _:
            main(main_args.as_script_args())

            # python = "python3"
            # # python = '/scratch2/NAGAPE/epic/Ben.Koziol/miniconda/envs/regrid-wrapper/bin/python3.11'
            # subprocess.check_call(
            #     [python, main_path] + list(
            #         main_args.as_script_args()))  # tdk: figure out python runtime


# class Test(unittest.TestCase):
#
#     def test_mask_edges(self) -> None:
#         #tdk:last: this is worth testing?
#         data = np.ma.array(np.ones((3,3)), mask=False)
#         mask_edges(data)
#         with open_nc("foo.nc", "w", parallel=False, clobber=True) as ds:
#             ds.createDimension('d', 3)
#             var = ds.createVariable('tester', 'f4', ('d', 'd'), fill_value=0.)
#             var[:] = data
#         with open_nc("foo.nc", "r", parallel=False) as ds:
#             actual = var[:]
#         assert actual.data.sum() == 1.
#
#     def test_enum_behaviors(self):
#
#         class Name(StrEnum):
#             ONE = 'one'
#             TWO = 'two'
#
#             def other_name(self):
#                 other = {self.ONE: 'foo-one', self.TWO: 'foo-two'}
#                 return other[self]
#
#         nn = Name('one')
#         assert nn.other_name() == 'foo-one'
#         assert list(Name) == [Name.ONE, Name.TWO]
#         assert list(Name) == ['one', 'two']
