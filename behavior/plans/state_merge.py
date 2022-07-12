import os
import shutil
import logging
import pandas as pd
from plumbum import cli
import numpy as np
from pathlib import Path
from behavior import OperatingUnit
from behavior.plans.utils import (
    process_time_pg_stats,
    process_time_pg_attribute,
    process_time_pg_class,
    process_time_pg_index,
    merge_modifytable_data,
    build_time_index_metadata,
    postgres_julian_to_unix
)

logger = logging.getLogger(__name__)


def process_time_pg_settings(time_pg_settings):
    time_pg_settings["unix_timestamp"] = time_pg_settings.time.astype(float) / 1e6
    time_pg_settings.reset_index(drop=True, inplace=True)
    time_pg_settings["pg_settings_identifier"] = time_pg_settings.index
    return time_pg_settings


def construct_all_time_metadatas(root):
    time_pg_attribute = process_time_pg_attribute(pd.read_csv(root / "pg_attribute.csv"))
    time_pg_index = process_time_pg_index(pd.read_csv(root / "pg_index.csv"))
    time_tables, time_cls_indexes = process_time_pg_class(pd.read_csv(root / "pg_class.csv"))
    time_index_metadata = build_time_index_metadata(time_pg_index, time_tables, time_cls_indexes, time_pg_attribute)

    # metadata should already be sorted on unix_timestamp
    time_index_metadata.sort_index(axis=0, inplace=True)
    time_index_metadata.indexrelid = time_index_metadata.indexrelid.astype(int)
    logger.info("Merged time_pg_index with pg_class and pg_attribute")

    time_pg_stats = process_time_pg_stats(pd.read_csv(root / "pg_stats.csv"))
    time_pg_stats.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    time_pg_stats.sort_index(axis=0, inplace=True)

    time_pg_settings = process_time_pg_settings(pd.read_csv(root / "pg_settings.csv"))
    time_pg_settings.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    time_pg_settings.sort_index(axis=0, inplace=True)
    return time_index_metadata, time_pg_settings, time_pg_stats


def merge_time_settings(data, settings):
    # sort by unix_timestamp.
    data.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    data.sort_index(axis=0, inplace=True)

    # left merge_asof. if not found, set to 0 which always provides a valid.
    data = pd.merge_asof(data, settings[["pg_settings_identifier"]], left_index=True, right_index=True, allow_exact_matches=True)
    data.fillna(value={"pg_settings_identifier":0}, inplace=True)
    data.reset_index(drop=False, inplace=True)
    return data


def merge_index_metadata(data, metadata, column, time_pg_stats):
    assert column in data.columns
    assert time_pg_stats.index.is_monotonic_increasing

    data.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    data.sort_index(axis=0, inplace=True)

    # left merge_asof.
    data = pd.merge_asof(data, metadata, left_index=True, right_index=True, left_by=[column], right_by=["indexrelid"], allow_exact_matches=True)

    # eliminate all rows with no valid metadata
    data.drop(data[data.indexrelid.isna()].index, inplace=True)
    data.reset_index(drop=False, inplace=True)

    # Switch index back to unix_timestamp.
    data.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    data.sort_index(axis=0, inplace=True)

    # These indicate the particular column stats that are interesting.
    indkey_atts = [key for key in data.columns if "indkey_attname_" in key]
    stats_view = time_pg_stats[["tablename", "attname", "pg_stats_identifier"]]
    for idx, indkey_att in enumerate(indkey_atts):
        left_by = ["table_relname", indkey_att]
        right_by = ["tablename", "attname"]
        data = pd.merge_asof(data, stats_view, left_index=True, right_index=True, left_by=left_by, right_by=right_by, allow_exact_matches=True)

        # Rename the key and drop the other useless columns.
        data.rename(columns={"pg_stats_identifier": f"indkey_statsid_{idx}"}, inplace=True)
        data.drop(labels=["tablename", "attname"], axis=1, inplace=True)

    data.reset_index(drop=False, inplace=True)
    return data


class StateMergeCLI(cli.Application):
    dir_datagen_diff = cli.SwitchAttr(
        "--dir-datagen-diff",
        Path,
        mandatory=True,
        help="Directory containing data that needs to be merged with snapshots.",
    )
    dir_output = cli.SwitchAttr(
        "--dir-output",
        Path,
        mandatory=True,
        help="Directory to output updated CSV files to.",
    )


    def process_generic(self, root, ou_file):
        logger.info("Processing %s", ou_file)
        data = pd.read_feather(root / f"{ou_file}.feather")
        data["data_identifier"] = data.index + 1
        data["unix_timestamp"] = postgres_julian_to_unix(data.statement_timestamp)
        return data


    def merge_index_metadata(self, root, ou_file, time_index_metadata, time_pg_settings, time_pg_stats, index_column):
        logger.info("Processing %s", ou_file)
        data = pd.read_feather(root / f"{ou_file}.feather")
        data["data_identifier"] = data.index + 1
        data["unix_timestamp"] = postgres_julian_to_unix(data.statement_timestamp)

        data = merge_time_settings(data, time_pg_settings)
        logger.info("Merged data with timestamped pg_settings identifier")

        data = merge_index_metadata(data, time_index_metadata, index_column, time_pg_stats)
        logger.info("Merged data with timestamped metadata and statistics")

        # Switch elapsed_us to the end of the dataframe (just for readability sakes).
        data.insert(len(data.columns) - 1, "elapsed_us", data.pop("elapsed_us"))
        return data


    def process_mt(self, root, name):
        logger.info("Processing %s", name)
        return merge_modifytable_data(name=name, root=root)


    def main(self):
        pd.options.display.width = 0
        pd.set_option('display.float_format', lambda x: '%.3f' % x)

        # Look through all the experiments.
        experiments = sorted(path.name for path in self.dir_datagen_diff.glob("*"))
        for experiment in experiments:
            # Look through all benchmarks within the experiment.
            experiment_root = self.dir_datagen_diff / experiment
            bench_names = sorted([d.name for d in experiment_root.iterdir() if d.is_dir()])
            for bench_name in bench_names:
                benchmark_path = f"{experiment_root}/{bench_name}/"
                logger.info("Processing benchmark %s", benchmark_path)

                merge_data_dir = self.dir_output / experiment / bench_name
                if merge_data_dir.exists():
                    shutil.rmtree(merge_data_dir)
                merge_data_dir.mkdir(parents=True, exist_ok=True)

                # Process all the index metadata upfront here.
                time_index_metadata, time_pg_settings, time_pg_stats = construct_all_time_metadatas(experiment_root / bench_name)

                # Start processing all the OUs.
                for f in Path(benchmark_path).glob("*.feather"):
                    try:
                        ou = OperatingUnit[f.stem]
                    except KeyError:
                        # Copy the feather not associated with an OU directly to the output.
                        shutil.copy(f, merge_data_dir / f"{f.stem}.feather")
                        continue

                    ou = OperatingUnit[f.stem]
                    if ou in [OperatingUnit.IndexScan, OperatingUnit.IndexOnlyScan, OperatingUnit.ModifyTableIndexInsert]:
                        column = {
                            OperatingUnit.IndexOnlyScan: "IndexOnlyScan_indexid",
                            OperatingUnit.IndexScan: "IndexScan_indexid",
                            OperatingUnit.ModifyTableIndexInsert: "ModifyTableIndexInsert_indexid"
                        }[ou]

                        data = self.merge_index_metadata(experiment_root / bench_name, f.stem, time_index_metadata, time_pg_settings, time_pg_stats, column)
                        logger.info("Writing out %s data", f.stem)
                        data.reset_index(drop=True, inplace=True)
                        groups = data.groupby(by=["relname"])
                        for group in groups:
                            logger.info("Splitting %s into %s", f.stem, group[0])
                            out = f"{merge_data_dir}/{f.stem}_{group[0]}.feather"
                            group[1].reset_index(drop=True).to_feather(out)
                    elif ou == OperatingUnit.ModifyTableInsert or ou == OperatingUnit.ModifyTableUpdate:
                        data = self.process_mt(experiment_root / bench_name, "ModifyTableInsert" if ou == OperatingUnit.ModifyTableInsert else "ModifyTableUpdate")
                        logger.info("Writing out %s data", f.stem)
                        data.reset_index(drop=True, inplace=True)
                        groups = data.groupby(by=["relname"])
                        for group in groups:
                            logger.info("Splitting %s into %s", f.stem, group[0])
                            out = f"{merge_data_dir}/{f.stem}_{group[0]}.feather"
                            group[1].reset_index(drop=True).to_feather(out)
                    else:
                        # Generically process
                        # Copy the file straight to the output directory.
                        # This will also copy any additional metadata-esque files.
                        data = self.process_generic(experiment_root / bench_name, f.stem)
                        logger.info("Writing out %s data", f.stem)
                        data.reset_index(drop=True, inplace=True)
                        data.to_feather(f"{merge_data_dir}/{f.stem}.feather")

                for f in Path(benchmark_path).glob("*.csv"):
                    if f.stem not in ["pg_attribute", "pg_index", "pg_stats", "pg_settings", "pg_trigger", "pg_constraint"]:
                        # Write all extra CSV files to the output directory.
                        shutil.copy(f, merge_data_dir / f"{f.stem}.csv")

                # Write out the processed pg_stats and settings for indexes.
                time_index_metadata.reset_index(drop=True, inplace=True)
                time_pg_settings.reset_index(drop=True, inplace=True)
                time_pg_stats.reset_index(drop=True, inplace=True)

                time_index_metadata.to_feather(f"{merge_data_dir}/idx_index_metadata.feather")
                time_pg_settings.to_feather(f"{merge_data_dir}/idx_pg_settings.feather")
                time_pg_stats.to_feather(f"{merge_data_dir}/idx_pg_stats.feather")

if __name__ == "__main__":
    StateMergeCLI.run()
