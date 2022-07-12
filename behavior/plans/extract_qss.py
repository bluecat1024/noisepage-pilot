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

from behavior import OperatingUnit, TARGET_COLUMNS, DERIVED_FEATURES_MAP
from behavior.plans import PLAN_INDEPENDENT_ID, UNIQUE_QUERY_ID_INDEX, QSS_PLANS_IGNORE_NODE_FEATURES, QSS_MERGE_PLAN_KEY

logger = logging.getLogger(__name__)


def prepare_qss_plans(plans_df):
    """
    Prepares the dataframe read from pg_qss_plans.csv for merging.
    Function returns <df, feature_dict> where df.plan_feature_idx indexes into feature_dict for full features.
        1. df.plan_feature_idx indexes into feature_dict for full features.
        2. df is sorted on statement_timestamp.
        3. df contains all columns in QSS_MERGE_PLAN_KEY
    """
    new_df_tuples = []
    new_plan_features = {}
    for index, row in tqdm(plans_df.iterrows(), total=plans_df.shape[0]):
        feature = json.loads(row.features)
        if len(feature) > 1:
            logger.warn("Skipping decoding of plan: %s", feature)
            continue

        def process_plan(plan):
            plan_features = {}
            for key, value in plan.items():
                if key in QSS_PLANS_IGNORE_NODE_FEATURES:
                    # Ignore the key-value feature.
                    continue

                if key == "Plans":
                    # For the special key, we recurse into the child.
                    for p in value:
                        process_plan(p)
                    continue

                if isinstance(value, list):
                    # TODO(wz2): For now, we simply featurize a list[] with a numeric length.
                    # This is likely insufficient if the content of the list matters significantly.
                    key = key + "_len"
                    value = len(value)

                plan_features[key] = value

            # Add this particular plan node's features to the feature dictionary.
            slot = len(new_plan_features) + 1
            new_plan_features[slot] = plan_features

            new_tuple = {
                'query_id': row.query_id,
                'db_id': row.db_id,
                'pid': row.pid,
                'statement_timestamp': row.statement_timestamp,
                'plan_node_id': plan["plan_node_id"],
                'plan_feature_idx': slot,
            }
            new_df_tuples.append(new_tuple)

        # For a given query, featurize each node in the plan tree.
        np_dict = feature[0]

        QUERY_SUBSTR_BLOCK = [
            "epoch from NOW()",
            "pg_prewarm",
            "version()",
            "current_schema()",
            "pg_settings",
            # The following are used to block JDBC.
            "pg_catalog.generate_series",
            "n.nspname = 'information_schema'",
            "pg_catalog.pg_namespace",
            # The following are used to block Benchbase.
            "pg_statio_user_indexes",
            "pg_stat_user_indexes",
            "pg_statio_user_tables",
            "pg_stat_user_tables",
            "pg_stat_database_conflicts",
            "pg_stat_database",
            "pg_stat_bgwriter",
            "pg_stat_archiver",
            "Pg_stat_archiver",
        ]

        skip = False
        query_text = np_dict["query_text"]
        for sub in QUERY_SUBSTR_BLOCK:
            if sub in query_text:
                # Block if the substring can be found.
                # This will then block all corresponding OUs.
                skip = True
                break

        if not skip:
            process_plan(np_dict)

    plans_df = pd.DataFrame(new_df_tuples)
    plans_df.set_index(keys=["statement_timestamp"], drop=True, append=False, inplace=True)
    plans_df.sort_index(axis=0, inplace=True)
    return plans_df, new_plan_features


def merge_ou_with_features(ou_df, plans_df, new_plan_features):
    """
    Combines the OU dataframe with relevant features from the query plan.
    Returns an augmented OU dataframe containing only valid matched OU entries.
    """

    assert np.sum(ou_df.plan_node_id == -1) == 0, "There should be no plan nodes that are -1."

    # (-2) is the PLAN_REMOTE_RECEIVER_ID.
    if np.sum(ou_df.plan_node_id <= PLAN_INDEPENDENT_ID) == len(ou_df):
        # All PLAN_INDENDENT_IDs start from (-3).
        return ou_df

    # Find the correct plan feature to use based on the timestamp.
    assert np.sum(ou_df.plan_node_id > PLAN_INDEPENDENT_ID) == len(ou_df)
    assert plans_df.index.is_monotonic_increasing
    ou_df.set_index(keys=["statement_timestamp"], drop=True, append=False, inplace=True)
    ou_df.sort_index(axis=0, inplace=True)
    ou_df = pd.merge_asof(ou_df, plans_df, left_index=True, right_index=True, by=["query_id", "db_id", "pid", "plan_node_id"], allow_exact_matches=True)

    # Drop all rows that fail to find a corresponding match.
    ou_df.drop(ou_df[ou_df.plan_feature_idx.isna()].index, inplace=True)
    if ou_df.shape[0] == 0:
        return ou_df

    ou_df.reset_index(drop=False, inplace=True)
    ou_df["id"] = ou_df.index

    def grab_features(row):
        # Yoink the corresponding plan node feature and add id to be used for joining.
        feature = new_plan_features[row.plan_feature_idx]
        feature['id'] = row.id
        return pd.Series(feature)

    # This produces a dataframe; we generally assume that any instance of a plan node will have the same features.
    features = ou_df.progress_apply(grab_features, axis=1, result_type='expand')
    features.set_index(keys=["id"], drop=True, append=False, inplace=True)
    ou_df.set_index(keys=["id"], drop=True, append=False, inplace=True)
    ou_df = ou_df.join(features, how="inner")

    # Normalize ou_df.
    ou_df.reset_index(drop=True, inplace=True)
    ou_df.drop(labels=["plan_feature_idx"], axis=1, inplace=True)
    return ou_df


def main(data_dir, output_dir, experiment, output_ous) -> None:
    logger.info("Extracting query state store features for experiment: %s", experiment)
    experiment_root: Path = data_dir / experiment
    bench_names: list[str] = [d.name for d in experiment_root.iterdir() if d.is_dir()]
    for bench_name in bench_names:
        logger.info("Benchmark: %s", bench_name)
        bench_root = experiment_root / bench_name
        ous_dir = bench_root / "ous"
        pg_qss_plans = bench_root / "pg_qss_plans.csv"
        assert pg_qss_plans.exists()
        if not ous_dir.exists():
            logger.info("Skipping %s", bench_name)
            continue

        # Process the query state store plans.
        plans_df, plan_features = prepare_qss_plans(pd.read_csv(pg_qss_plans))

        extract_data_dir: Path = output_dir / experiment / bench_name
        if extract_data_dir.exists():
            shutil.rmtree(extract_data_dir)
        extract_data_dir.mkdir(parents=True, exist_ok=True)

        for csv_file in bench_root.glob("*.csv"):
            if csv_file.stem == "pg_qss_stats":
                continue
            # These are non-execution OU data. These files are copied out
            # for something downstream to consume.
            shutil.copy(csv_file, extract_data_dir / f"{csv_file.stem}.csv")

        # This computes the actual OUs that we need to process and output.
        output_ous = set([node_name.name for node_name in OperatingUnit]).intersection(set(output_ous))
        exec_ous = sorted([(node_name, ous_dir / f"Exec{node_name}.feather") for node_name in output_ous], key=lambda x: x[0])
        for node_name, ou in exec_ous:
            if not ou.exists():
                # If the OU data does not exist, then it is a no-op.
                continue

            logger.info("Processing %s", ou)
            ou_df = pd.read_feather(ou)
            if ou_df.shape[0] == 0:
                # If the OU has no data, then it is a no-op.
                continue

            # Combine the node feature data with the actual plan node execution.
            ou_df = merge_ou_with_features(ou_df, plans_df, plan_features)
            if ou_df.shape[0] == 0:
                logger.info("%s does not have any valid plan data.", ou)
                continue

            # Reset index before writing to feather file.
            ou_df.reset_index(drop=True, inplace=True)
            ou_df.to_feather(extract_data_dir / f"{node_name}.feather")


class ExtractQSSCLI(cli.Application):
    dir_datagen_data = cli.SwitchAttr(
        "--dir-datagen-data",
        Path,
        mandatory=True,
        help="Directory containing DataGenerator output data.",
    )
    dir_output = cli.SwitchAttr(
        "--dir-output",
        Path,
        mandatory=True,
        help="Directory to output updated feather files to.",
    )
    glob_pattern = cli.SwitchAttr(
        "--glob-pattern", mandatory=False, help="Glob pattern to use for selecting valid experiments."
    )
    output_ous = cli.SwitchAttr(
        "--output-ous",
        mandatory=False,
        default=",".join([e.name for e in OperatingUnit]),
        help="List of OUs to output that are comma separated."
    )

    def main(self):
        tqdm.pandas()
        train_folder = self.dir_datagen_data

        # By default, difference all the valid experiments.
        pattern = "*" if self.glob_pattern is None else self.glob_pattern
        experiments = sorted(path.name for path in train_folder.glob(pattern))
        assert len(experiments) > 0, "No training data found?"

        output_ous = self.output_ous.split(",")
        for experiment in experiments:
            main(self.dir_datagen_data, self.dir_output, experiment, output_ous)


if __name__ == "__main__":
    ExtractQSSCLI.run()
