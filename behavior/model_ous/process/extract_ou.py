from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from plumbum import cli
from tqdm import tqdm

from behavior import DERIVED_FEATURES_MAP

logger = logging.getLogger(__name__)


def transform_ou_df(node_name, ou_df):
    """
    Given the current OU dictated by [node_name], perform the correct transformation to ou_df.
    This function drops and/or renames columns as needed based on the OU.

    This is mainly for readability and understandability later in the pipeline but can also
    be a point to perform certain transformations.
    """
    derived_map = {}
    for k, v in DERIVED_FEATURES_MAP.items():
        if k.startswith(node_name):
            derived_map[v] = k

    if len(derived_map) > 0:
        ou_df.rename(columns=derived_map, inplace=True, errors='raise')

    # IndexScan_num_outer_loops should be at least 1.
    # There are recording cases where this value can be 0 so we adjust it.
    if "IndexScan_num_outer_loops" in ou_df.columns:
        ou_df["IndexScan_num_outer_loops"] = np.clip(ou_df.IndexScan_num_outer_loops, 1, None)

    features_drop = [f"counter{i}" for i in range(10)] + ["payload", "comment", "txn"]
    ou_df.drop(columns=features_drop, inplace=True, errors='ignore')
    return ou_df


def main(data_dir, experiment) -> None:
    logger.info("Extracting OU features for experiment: %s", experiment)
    experiment_root: Path = data_dir / experiment
    bench_names: list[str] = [d.name for d in experiment_root.iterdir() if d.is_dir()]
    for bench_name in bench_names:
        logger.info("Benchmark: %s", bench_name)
        bench_root = experiment_root / bench_name
        pg_qss_stats = bench_root / "pg_qss_stats.csv"
        assert pg_qss_stats.exists()

        # Load in the query state store execution counters.
        qss_stats = pd.read_csv(pg_qss_stats)
        qss_stats = qss_stats[(qss_stats.plan_node_id != -1) & (qss_stats.query_id != 0) & (qss_stats.statement_timestamp != 0)]
        if qss_stats.shape[0] == 0:
            logger.info("Skipping %s", bench_name)
            continue

        ous = bench_root / "ous"
        ous.mkdir(parents=True, exist_ok=True)

        ou_groups = qss_stats.groupby(["comment"])
        for ou_group in ou_groups:
            output_file = ous / f"Exec{ou_group[0]}.feather"
            ou = transform_ou_df(ou_group[0], ou_group[1])
            ou.reset_index(drop=True, inplace=True)
            ou.to_feather(output_file)


class ExtractOUCLI(cli.Application):
    dir_datagen_data = cli.SwitchAttr(
        "--dir-datagen-data",
        Path,
        mandatory=True,
        help="Directory containing DataGenerator output data.",
    )
    glob_pattern = cli.SwitchAttr(
        "--glob-pattern", mandatory=False, help="Glob pattern to use for selecting valid experiments."
    )

    def main(self):
        tqdm.pandas()
        train_folder = self.dir_datagen_data

        # By default, difference all the valid experiments.
        pattern = "*" if self.glob_pattern is None else self.glob_pattern
        experiments = sorted(path.name for path in train_folder.glob(pattern))
        assert len(experiments) > 0, "No training data found?"

        for experiment in experiments:
            main(self.dir_datagen_data, experiment)


if __name__ == "__main__":
    ExtractOUCLI.run()
