from pathlib import Path

from ads.core import Context, S3SyncRunner, UseCase, UseCaseKey, UseCaseAeromma


class TestS3SyncRunner:

    def test_happy_path(self, tmp_path: Path) -> None:
        first_cycle_date = "2023060112"
        ctx = Context(first_cycle_date=first_cycle_date, dst_dir=tmp_path, dry_run=True)
        runner = S3SyncRunner(ctx)
        runner.run()

    def test_create_sync_command(self, tmp_path: Path) -> None:
        first_cycle_date = "2023060112"
        last_cycle_date = "2023060212"
        dst_dir = tmp_path / "output-for-this-test"
        ctx = Context(
            first_cycle_date=first_cycle_date,
            last_cycle_date=last_cycle_date,
            dst_dir=dst_dir,
            dry_run=True,
        )
        runner = S3SyncRunner(ctx)
        actual = runner._create_sync_cmd_()
        expected = (
            "aws",
            "s3",
            "sync",
            "--no-sign-request",
            "--dryrun",
            "--exclude",
            "*",
            "--include",
            "FV3GFS/gfs.20230601/12/atmos/gfs.t12z.atmf000.nc",
            "--include",
            "GFS_SFC_DATA/gfs.20230601/12/atmos/gfs.sfcanl.nc",
            "--include",
            "GFS_SFC_DATA/gfs.20230601/12/atmos/gfs.t12z.sfcf000.nc",
            "--include",
            "GEFS_Aerosol/20230601/00/gfs.t00z.atmf000.nemsio",
            "--include",
            "RAVE_fire/rave-20230601.tar",
            "--include",
            "RESTART/*20230531*",
            "--include",
            "FV3GFS/gfs.20230602/12/atmos/gfs.t12z.atmf000.nc",
            "--include",
            "GFS_SFC_DATA/gfs.20230602/12/atmos/gfs.sfcanl.nc",
            "--include",
            "GFS_SFC_DATA/gfs.20230602/12/atmos/gfs.t12z.sfcf000.nc",
            "--include",
            "GEFS_Aerosol/20230602/00/gfs.t00z.atmf000.nemsio",
            "--include",
            "RAVE_fire/rave-20230602.tar",
            "s3://noaa-ufs-srw-pds/UFS-AQM",
            str(dst_dir),
        )
        assert actual == expected


class TestUseCase:

    def test_use_case(self, tmp_path: Path) -> None:
        use_case = UseCase.from_key(UseCaseKey.AEROMMA, dst_dir=tmp_path)
        print(use_case)
        assert isinstance(use_case, UseCaseAeromma)
