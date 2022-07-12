from __future__ import annotations

import itertools
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from pandas import DataFrame
from plumbum import cli
from sklearn import tree

from behavior import TARGET_COLUMNS, OperatingUnit, Targets, DERIVED_FEATURES_MAP
from behavior.modeling import featurize
from behavior.modeling.utils.prepare_data import load_input_data
from behavior.modeling.utils.evaluate_ou import evaluate_ou_model
from behavior.modeling.model import BehaviorModel

logger = logging.getLogger(__name__)

IGNORE_COLS = [
    # Ignore all columns that are identifying information.
    "data_identifier",
    "source_file",
]


def load_data(train_files):
    """Load the training data.

    Parameters
    ----------
    train_files: List[Path]
        List of files to consider for training data.

    Returns
    -------
    dict[str, DataFrame]
        A map from operating unit names to their training data.

    Raises
    ------
    Exception
        If there is no valid training data.
    """
    # Load all the OU data from disk given the data directory.
    # We filter all files with zero results because it is common to only have data for
    # a few operating units.
    result_paths = [fp for fp in train_files if os.stat(fp).st_size > 0]
    ou_name_to_df: dict[str, DataFrame] = {}
    for ou in OperatingUnit:
        ou_name = ou.name
        ou_results = [fp for fp in result_paths if fp.name.startswith(ou_name)]
        if len(ou_results) > 0:
            logger.debug("Found %s run(s) for %s", len(ou_results), ou_name)
            def invoke(path):
                # We are loading data for training purposes.
                return load_input_data(logger, path, {}, True)

            ou_name_to_df[ou_name] = pd.concat(map(invoke, ou_results))

    # We should always have data for at least one operating unit.
    if len(ou_name_to_df) == 0:
        raise Exception(f"No data found in data_dirs: {data_dirs}")

    return ou_name_to_df


def main(
    config_file,
    dir_data,
    dir_output,
    use_featurewiz,
    prefix_allow_derived_features,
):
    # Load modeling configuration.
    if not config_file.exists():
        raise ValueError(f"Config file: {config_file} does not exist")

    logger.info("Loading config file: %s", config_file)
    with config_file.open("r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["modeling"]

    # Block derived features
    prefix_allows = prefix_allow_derived_features.split(",")
    for n in DERIVED_FEATURES_MAP.keys():
        allow = False
        for prefix_allow in prefix_allows:
            if n.startswith(prefix_allow):
                allow = True

        if not allow:
            IGNORE_COLS.append(n)

    # Mark this training-evaluation run with a timestamp for identification.
    training_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_files = list(dir_data.rglob("*.feather"))
    assert len(train_files) > 0, "No matching data files for training could be found."

    # Load the data and name the model.
    train_ou_to_df = load_data(train_files)
    output_dir = dir_output / f"{config_file.stem}_{training_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "source.txt").open("w+") as f:
        f.write(f"Train: {train_files}\n")

    for ou_name, df_train in train_ou_to_df.items():
        logger.info("Begin Training OU: %s", ou_name)
        logger.info("Deriving input features for OU: %s", ou_name)
        targets = [Targets.ELAPSED_US.value]
        if use_featurewiz:
            # TODO(wz2): Currently prioritize the `elapsed_us` target. We may want to specify multiple
            # targets in the future and/or consider training a model for each target separately.
            features = featurize.derive_input_features(
                df_train,
                feature_engg=None,
                ignore=IGNORE_COLS,
                test=None,
                targets=targets,
                config=config["featurize"]
            )
        else:
            # Get a metadata representation for extracting all feature columns.
            features = featurize.extract_all_features(df_train, ignore=IGNORE_COLS)

        # Partition the features and targets.
        x_train = featurize.extract_input_features(df_train, features)
        y_train = df_train[targets].values

        # Check if no valid training data was found (for the current operating unit).
        if x_train.shape[1] == 0 or y_train.shape[1] == 0:
            logger.warning(
                "OU: %s has no valid training data, skipping. Feature cols: %s, X_train shape: %s, y_train shape: %s",
                ou_name,
                features,
                x_train.shape,
                y_train.shape,
            )
            continue

        # Train one model for each method specified in the modeling configuration.
        for method in config["methods"]:
            logger.info("Training OU: %s with model: %s", ou_name, method)
            ou_model = BehaviorModel(
                method,
                ou_name,
                config,
                features,
                targets=targets,
            )

            # Train and save the model.
            output = output_dir / method
            output.mkdir(parents=True, exist_ok=True)

            ou_model.train(x_train, y_train)
            ou_model.save(output)
            evaluate_ou_model(ou_model, output, "train", eval_file=None, eval_df=df_train)


class TrainCLI(cli.Application):
    config_file = cli.SwitchAttr(
        "--config-file",
        Path,
        mandatory=True,
        help="Path to configuration YAML containing modeling parameters.",
    )
    dir_data = cli.SwitchAttr(
        "--dir-data",
        Path,
        mandatory=True,
        help="Root of a recursive glob to gather all feather files for training data.",
    )
    dir_output = cli.SwitchAttr(
        "--dir-output",
        Path,
        mandatory=True,
        help="Folder to output models to.",
    )
    use_featurewiz = cli.Flag(
        "--use-featurewiz",
        help="Whether to use featurewiz for feature selection.",
    )
    prefix_allow_derived_features = cli.SwitchAttr(
        "--prefix-allow-derived-features",
        help="List of prefixes to use for selecting derived features for training the model.",
        default=""
    )

    def main(self):
        main(
            self.config_file,
            self.dir_data,
            self.dir_output,
            self.use_featurewiz,
            self.prefix_allow_derived_features,
        )


if __name__ == "__main__":
    TrainCLI.run()
