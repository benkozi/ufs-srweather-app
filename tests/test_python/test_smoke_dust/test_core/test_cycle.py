from pathlib import Path

import pytest
from _pytest.fixtures import SubRequest

from smoke_dust.core.context import SmokeDustContext
from smoke_dust.core.cycle import SmokeDustCycleTwo
from smoke_dust.core.preprocessor import SmokeDustPreprocessor
from test_python.test_smoke_dust.conftest import create_fake_context, create_fake_grid_out, \
    FakeGridOutShape


@pytest.fixture(params=[True, False], ids= lambda p: f"allow_dummy_restart={p}")
def context_for_test(request: SubRequest, tmp_path: Path, fake_grid_out_shape: FakeGridOutShape) -> SmokeDustContext:
    create_fake_grid_out(tmp_path, fake_grid_out_shape)
    context = create_fake_context(tmp_path, overrides={"allow_dummy_restart": request.param})
    return context


def test_writes_dummy_emissions_with_no_restart_files(context_for_test: SmokeDustContext) -> None:
    preprocessor = SmokeDustPreprocessor(context_for_test)
    cycle = SmokeDustCycleTwo(context_for_test)
    assert not context_for_test.emissions_path.exists()
    try:
        cycle.run(preprocessor.forecast_metadata)
    except FileNotFoundError:
        assert not context_for_test.allow_dummy_restart
    else:
        assert context_for_test.emissions_path.exists()
