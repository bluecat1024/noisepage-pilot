from __future__ import annotations

import gc
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from plumbum import cli
from tqdm import tqdm

from behavior import OperatingUnit
from behavior.model_ous.process import (
    DiffPlanIncompleteSubinvocationException,
    DiffPlanInvalidDataException,
    DiffPlanUnsupportedParallelException,

    UNIQUE_QUERY_ID_INDEX,
    DIFF_BLOCKED_OUS,
    DIFF_SKIP_OUS,
    DIFF_SCHEMA_WITH_TARGETS,
)

logger = logging.getLogger(__name__)

def load_feather(ou_index, feather_file):
    """
    Read the feather file for a given OU.

    Parameters
    ----------
    ou_index : int
        Index into OperatingUnit that describes the particular OU this feather file
        corresponds to.

    feather_file: Path
        Path to the feather file that should be read.

    Returns
    -------
    ou_index : int
        Index of the OperatingUnit that the dataframe corresponds to.
    df_targets : pd.DataFrame
        Dataframe constructed from the feather file for differencing.
    df_features : pd.DataFrame
        Dataframe constructed from the feather file of non-differencing columns.

    Notes
    -----
        - Both `df_targets` and `df_features` contain `ou_index` and `data_id`. These features are used for joining the two dataframes.
        - `df_targets` follows the DIFF_SCHEMA_WITH_TARGETS layout
    """

    # Read the CSV file and add `ou_index` and `data_id`.
    df = pd.read_feather(feather_file)
    df["ou_index"] = ou_index
    df["data_id"] = df.index

    # Here we produce two different dataframes from the original feather file.
    #
    # df[targets] produces a dataframe containing all the common schema columns and the
    # target columns. Plan differencing is performed on this dataframe.
    #
    # df[features] contains all the other columns in the input data that were not used
    # for differencing along with `ou_index` and `data_id`. It is worth noting that
    # `ou_index` and `data_id` can be used to reconstruct a datapoint across the
    # two dataframes.
    targets = DIFF_SCHEMA_WITH_TARGETS
    features = ["ou_index", "data_id"] + list(set(df.columns) - set(targets))

    # pylint: disable=E1136
    return ou_index, df[targets], df[features]


def load_noisepage_data(data_dir):
    """
    Load NoisePage data into dataframes.

    Parameters
    ----------
    data_dir : Path
        Data directory containing all the NoisePage data with features.

    Returns
    --------
    unified : pd.DataFrame
         Dataframe containing datapoints from all OUs arranged by DIFF_SCHEMA_WITH_TARGETS.

    features : dict[int, pd.DataFrame]
         Dictionary mapping a ou_index (index in OperatingUnit) to OU specific features.
    """

    result_paths = {
        i: data_dir / f"{node_name.name}.feather"
        for i, node_name in enumerate(OperatingUnit)
        if node_name not in DIFF_BLOCKED_OUS
        and node_name not in DIFF_SKIP_OUS
    }

    result_paths = {
        key: value for (key, value) in result_paths.items() if value.exists() and os.stat(value).st_size > 0
    }

    extra_files = set([f for f in data_dir.glob("*.feather") if OperatingUnit[f.stem] not in DIFF_BLOCKED_OUS]) - set(result_paths.values())
    extra_files.update([f for f in data_dir.glob("*.csv")])

    ou_indexes, commons, features = zip(*[load_feather(key, value) for (key, value) in result_paths.items()])
    features = dict(zip(ou_indexes, features))
    unified = pd.concat(commons, axis=0, copy=False)
    return unified, features, extra_files


def diff_query_invocation(subinvocation, diffed_matrices):
    """
    Diffs a given query invocation by calling a CPython function.

    Parameters
    ----------
    subinvocation : pd.DataFrame
        Dataframe describing a single invocation of an unique query instance.

    diffed_matrices : list[np.pdarray]
        Output list to store diffed numpy matrices.
    """
    # diff_c.so is compiled from behavior/model_ous/process/diff_c.pyx. If an error is thrown
    # saying that diff_c can't be found, then please ensure your PYTHONPATH is
    # setup correctly.
    #
    # pylint: disable=E0401,C0415
    from diff_c import diff_query_tree

    # The 2D underlying subinvocation array is cast to a float64[][] for efficient Cython
    # indexing into numpy ndarrays.
    matrix = subinvocation.to_numpy(dtype=np.float64, copy=False)

    try:
        diff_query_tree(matrix)
    except (DiffPlanInvalidDataException, DiffPlanIncompleteSubinvocationException) as e:
        print("Invalid Data detected for subinvocation", subinvocation, matrix)
        raise e
    except DiffPlanUnsupportedParallelException:
        # These are not fatal errors. In these cases, we just return None to indicate
        # that there is no data that needs to be merged.
        return None

    # Append the diffed numpy array to diffed_matrices for us to post-process at once.
    diffed_matrices.append(matrix)
    return None


def process_query_invocation(subframe, diffed_matrices):
    """
    Function used to difference all data associated with a given query session template.

    Parameters
    ----------
    subframe : pd.DataFrame
        Dataframe contains the data that we want to difference. The dataframe must be
        data that is associated with a given query session template.

        In other words, UNIQUE_QUERY_ID_INDEX is the same for all rows.

    diffed_matrices : list[np.pdarray]
        Output list to store diffed numpy matrices.
    """
    assert (subframe["plan_node_id"] == 0).sum() == 1
    diff_query_invocation(subframe, diffed_matrices=diffed_matrices)
    return None


def diff_queries(unified, diffed_matrices):
    """
    Diff all queries in the input data.

    Parameters
    ----------
    unified : pd.DataFrame
        Dataframe contains all the data that needs to be diferenced. The dataframe must folow
        the DIFF_SCHEMA_WITH_TARGETS layout.

    diffed_matrices : list[np.pdarray]
        Output list to store diffed numpy matrices.
    """
    # Here we assume that UNIQUE_QUERY_ID_INDEX identifies a unique query session template.
    # Grouping by UNIQUE_QUERY_ID_INDEX will produce subframes of OUs for a given query session template.
    invocation_groups = unified.groupby(by=UNIQUE_QUERY_ID_INDEX, sort=False)

    # We use apply() because we want to process the entire subframe as a single unit. transform() will not work
    # for this because transform() may separate the columns.
    invocation_groups.progress_apply(process_query_invocation, diffed_matrices=diffed_matrices)


def save_results(diff_data_dir, ou_to_features, unified, output_ous, extra_files):
    """
    Save the new dataframes to disk.

    Parameters
    ----------
    diff_data_dir : Path
        Directory to save the differenced data.

    ou_to_features : dict[ou_index, DataFrame]
        Map from index indicating OU to OU specific features.

    unified : DataFrame
        DataFrame of all differenced records with an index on <ou_index, data_id>.

    extra_files : List[Path]
        List of extra files to copy to output directory.
    """

    _ = [df.set_index(["ou_index", "data_id"], drop=True, inplace=True) for (_, df) in ou_to_features.items()]
    for ou_index, features in tqdm(ou_to_features.items()):
        if OperatingUnit(ou_index).name not in output_ous:
            # Don't output an OU that we don't need
            continue

        # Perform an inner join with the index (on=None) of `ou_index` and `data_id`.
        # No suffixes are specified since features and unified should not share columns.
        result = features.join(unified, on=None, how="inner")
        if result.shape[0] > 0:
            cols = unified.columns.tolist() + features.columns.tolist()
            result = result[cols]

            # If we find that there are matching output rows, write them out.
            # Don't write out the index columns.
            result.reset_index(drop=True, inplace=True)
            result.to_feather(f"{diff_data_dir}/{OperatingUnit(ou_index).name}.feather")

    for extra_file in extra_files:
        shutil.copy(extra_file, f"{diff_data_dir}/{extra_file.stem}{extra_file.suffix}")


def main(data_dir, output_dir, experiment, output_ous) -> None:
    logger.info("Differencing experiment: %s", experiment)
    experiment_root: Path = data_dir / experiment
    bench_names: list[str] = [d.name for d in experiment_root.iterdir() if d.is_dir()]

    for bench_name in bench_names:
        logger.info("Benchmark: %s", bench_name)
        bench_root = experiment_root / bench_name
        diff_data_dir: Path = output_dir / experiment / bench_name
        if diff_data_dir.exists():
            shutil.rmtree(diff_data_dir)
        diff_data_dir.mkdir(parents=True, exist_ok=True)

        # Do this so we apply the proper schema normalization and such.
        unified, features, extra_files = load_noisepage_data(bench_root)

        # Reset the index on unified and default initialize the subinvocation_id field.
        unified.reset_index(drop=True, inplace=True)
        diffed_matrices: list[np.ndarray] = []
        diff_queries(unified, diffed_matrices)

        # Concatenate diffed_matrices back into a single dataframe.
        unified_np = np.concatenate(diffed_matrices, axis=0)
        unified_replace = DataFrame(unified_np, copy=False, columns=unified.columns)

        # Replace the following columns from unified -> unified_replace.
        labels_replace = ["query_id", "db_id", "statement_timestamp"]
        unified_replace.set_index(keys=["ou_index", "data_id"], drop=True, inplace=True)
        unified.set_index(keys=["ou_index", "data_id"], drop=True, inplace=True)
        unified_replace.drop(labels=labels_replace, axis=1, inplace=True)
        unified.drop(labels=[col for col in unified.columns if col not in labels_replace], axis=1, inplace=True)
        unified = unified_replace.join(unified, how="inner")
        del unified_np
        del unified_replace
        del diffed_matrices
        gc.collect()
        gc.collect()

        # Write the results out to the output directory.
        save_results(diff_data_dir, features, unified, output_ous, extra_files)
        del unified
        del features
        gc.collect()
        gc.collect()


class DiffCLI(cli.Application):
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
        help="Directory to output differenced feather files to.",
    )
    glob_pattern = cli.SwitchAttr(
        "--glob-pattern", mandatory=False, help="Glob pattern to use for selecting valid experiments."
    )
    output_ous = cli.SwitchAttr(
        "--output-ous",
        mandatory=False,
        default=",".join([node.name for node in OperatingUnit]),
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
            main(self.dir_datagen_data,
                 self.dir_output,
                 experiment,
                 output_ous)


if __name__ == "__main__":
    DiffCLI.run()
