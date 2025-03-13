from pathlib import Path

import pandas as pd
import xarray as xr
from pydantic import BaseModel


class DescribeParams(BaseModel):
    namespace: str
    files: tuple[Path, ...]
    varnames: tuple[str, ...]
    csv_out: Path | None = None


def describe(params: DescribeParams) -> pd.DataFrame:
    summary = []
    for idx, f in enumerate(params.files):
        with xr.open_dataset(f) as ds:
            print(f"{idx + 1} of {len(params.files)}: {f=}")
            for varname in params.varnames:
                print(varname)
                row = {"file": f, "namespace": params.namespace}
                row["varname"] = varname
                row["median"] = ds[varname].median().values.item()
                row["min"] = ds[varname].min().values.item()
                row["max"] = ds[varname].max().values.item()
                row["mean"] = ds[varname].mean().values.item()
                row["std"] = ds[varname].std().values.item()
                for q in [0.25, 0.5, 0.75]:
                    row[f"{q:.0%}"] = ds[varname].quantile(q).values.item()
                row["sum"] = ds[varname].sum().values.item()
                row["n"] = ds[varname].count().values.item()
                row["n_null"] = ds[varname].isnull().sum().values.item()
                summary.append(row)
    df = pd.DataFrame(summary)
    if params.csv_out:
        df.to_csv(params.csv_out, index=False)
    return df
