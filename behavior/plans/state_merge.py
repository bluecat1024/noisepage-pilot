import os
import shutil
import logging
import pandas as pd
from plumbum import cli
import numpy as np
from pathlib import Path
from behavior import OperatingUnit

logger = logging.getLogger(__name__)


def postgres_julian_to_unix(df):
    # Formula to convert postgres JULIAN to unix timestamp.
    return (df / float(1e6)) + 86400 * (2451545 - 2440588)


def process_time_pg_attribute(time_pg_attribute):
    PG_ATTRIBUTE_SCHEMA = [
        "attrelid",
        "attname",
        "atttypid",
        # "attstattarget",
        "attlen",
        "attnum",
        # "attndims",
        # "attcacheoff",
        "atttypmod",
        # "attbyval",
        # "attalign",
        "attstorage",
        "attcompression",
        "attnotnull",
        # "atthasdef",
        # "atthasmissing",
        # "attidentity",
        # "attgenerated",
        "attisdropped",
        # "attislocal",
        # "attinhcount",
        # "attcollation",
        # "attacl",
        # "attoptions",
        # "attfdwoptions",
        # "attmissingval",
    ]

    cols_remove = [col for col in time_pg_attribute.columns if col not in PG_ATTRIBUTE_SCHEMA]
    time_pg_attribute["unix_timestamp"] = time_pg_attribute.time.astype(float) / 1e6
    time_pg_attribute.drop(labels=cols_remove, axis=1, inplace=True, errors='ignore')
    return time_pg_attribute


def process_time_pg_class(time_pg_class):
    PG_CLASS_SCHEMA = [
        "time",
        "oid",
        "relname",
        # "relnamespace",
        # "reltype",
        # "reloftype",
        # "relowner",
        # "relam",
        # "relfilenode",
        # "reltablespace",
        "relpages",
        "reltuples",
        "relallvisible",
        "reltoastrelid",
        # "relhasindex",
        # "relisshared",
        # "relpersistence",
        "relkind",
        "relnatts",
        # "relchecks",
        # "relhasrules",
        # "relrelhastriggers",
        # "relhassubclass",
        # "relrowsecurity",
        # "relforcerowsecurity",
        # "relispopulated",
        # "relreplident",
        # "relispartition",
        # "relrewrite",
        # "relfroxzenxid",
        # "relminmxid",
        # "relacl",
        # "reloptions",
        # "relpartbound",
    ]

    cols_remove = [col for col in time_pg_class.columns if col not in PG_CLASS_SCHEMA]
    time_pg_class.drop(labels=cols_remove, axis=1, inplace=True, errors='ignore')
    time_pg_class["unix_timestamp"] = time_pg_class.time.astype(float) / 1e6
    time_pg_class.reset_index(drop=True, inplace=True)
    time_pg_class.drop(labels=["time"], axis=1, inplace=True, errors='ignore')

    tables = time_pg_class[(time_pg_class.relkind == 'r') | (time_pg_class.relkind == 't')]
    indexes = time_pg_class[time_pg_class.relkind == 'i']
    indexes = indexes.drop(labels=["relallvisible", "reltoastrelid", "relkind"], axis=1, errors='ignore')

    tables.reset_index(drop=True, inplace=True)
    indexes.reset_index(drop=True, inplace=True)
    return tables, indexes


def process_time_pg_index(time_pg_index):
    PG_INDEX_SCHEMA = [
        "indexrelid",
        "indrelid",
        "indnatts",
        "indnkeyatts",
        "indisunique",
        "indisprimary",
        "indisexclusion",
        "indimmediate",
        # "indisclustered",
        # "indisvalid",
        # "indcheckxmin",
        # "indisready",
        # "indislive",
        # "indisreplident",
        "indkey",
        # "indcollation",
        # "indclass",
        # "indoption",
        # "indexprs",
        # "indpred",
    ]

    cols_remove = [col for col in time_pg_index.columns if col not in PG_INDEX_SCHEMA]
    time_pg_index["unix_timestamp"] = time_pg_index.time.astype(float) / 1e6
    time_pg_index.drop(labels=cols_remove, axis=1, inplace=True, errors='ignore')
    return time_pg_index


def process_time_pg_stats(time_pg_stats):
    PG_STATS_SCHEMA = [
        "time",
        "schemaname",
        "tablename",
        "attname",
        # "inherited",
        "null_frac",
        "avg_width",
        "n_distinct",
        "most_common_vals",
        "most_common_freqs",
        "histogram_bounds",
        "correlation",
        # "most_common_elems",
        # "most_common_elem_freqs",
        # "elem_count_histogram",
    ]

    cols_remove = [col for col in time_pg_stats.columns if col not in PG_STATS_SCHEMA]
    time_pg_stats.drop(labels=cols_remove, axis=1, inplace=True, errors='ignore')

    public_mask = (time_pg_stats.schemaname != 'public')
    time_pg_stats.drop(time_pg_stats[public_mask].index, axis=0, inplace=True)
    time_pg_stats.drop(labels=["schemaname"], axis=1, inplace=True, errors='ignore')
    time_pg_stats["unix_timestamp"] = time_pg_stats.time.astype(float) / 1e6
    time_pg_stats.reset_index(drop=True, inplace=True)
    time_pg_stats["pg_stats_identifier"] = time_pg_stats.index
    return time_pg_stats


def process_time_pg_settings(time_pg_settings):
    time_pg_settings["unix_timestamp"] = time_pg_settings.time.astype(float) / 1e6
    time_pg_settings.reset_index(drop=True, inplace=True)
    time_pg_settings["pg_settings_identifier"] = time_pg_settings.index
    return time_pg_settings


def build_time_index_metadata(pg_index, tables, cls_indexes, pg_attribute):
    # First order of attack is to join cls_indexes against pg_index.
    cls_indexes.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    cls_indexes.sort_index(axis=0, inplace=True)
    pg_index.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    pg_index.sort_index(axis=0, inplace=True)

    time_pg_index = pd.merge_asof(cls_indexes, pg_index, left_index=True, right_index=True, left_by=["oid"], right_by=["indexrelid"], allow_exact_matches=True)
    time_pg_index.drop(time_pg_index[time_pg_index.indexrelid.isna()].index, inplace=True)

    # It's super unfortunate that we have to do this but this is because merge_asof can
    # insert NaN which converts an integer column -> float column. so we convert it back :)
    time_pg_index.indrelid = time_pg_index.indrelid.astype(int)
    del cls_indexes
    del pg_index

    # Second order of attack is to join time_cls_indexes to tables (tables in pg_class).
    time_pg_index.sort_index(axis=0, inplace=True)
    tables.columns = "table_" + tables.columns
    tables.set_index(keys=["table_unix_timestamp"], drop=True, append=False, inplace=True)
    tables.sort_index(axis=0, inplace=True)

    time_pg_index = pd.merge_asof(time_pg_index, tables, left_index=True, right_index=True, left_by=["indrelid"], right_by=["table_oid"], allow_exact_matches=True)
    time_pg_index.drop(time_pg_index[time_pg_index.table_oid.isna()].index, inplace=True)
    time_pg_index.drop(labels=["table_oid"], axis=1, inplace=True)
    time_pg_index.indrelid = time_pg_index.indrelid.astype(int)
    del tables

    # Third order of attack is to extract the indkey
    SENTINEL = 0
    indkeys = time_pg_index.indkey.str.split(" ", expand=True)
    assert not ((indkeys == SENTINEL).any().any())
    indkeys.fillna(value=SENTINEL, inplace=True)
    indkeys = indkeys.astype(int)
    indkeys_cols = [f"indkey_slot_{i}" for i in indkeys.columns]
    time_pg_index[indkeys_cols] = indkeys
    time_pg_index.drop(labels=["indkey"], axis=1, inplace=True)

    keep_cols = ["attrelid", "attnum", "attname", "attlen", "atttypmod"]
    pg_attribute.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    pg_attribute.drop(labels=[col for col in pg_attribute if col not in keep_cols], axis=1, inplace=True)

    time_pg_index.sort_index(axis=0, inplace=True)
    for i, slot in enumerate(indkeys_cols):
        time_pg_index = pd.merge_asof(time_pg_index, pg_attribute, left_index=True, right_index=True, left_by=["indrelid", slot], right_by=["attrelid", "attnum"], allow_exact_matches=True)

        # Drop all rows where we expect an attribute to be present but there is None.
        mask = time_pg_index[slot].isna() & time_pg_index.attname.isna()
        time_pg_index.drop(time_pg_index[mask].index, inplace=True)

        # Rename to produce unique column names.
        time_pg_index.rename(columns={"attname": f"indkey_attname_{i}", "attlen": f"indkey_attlen_{i}", "atttypmod": f"indkey_atttypmod_{i}"}, inplace=True)
        time_pg_index.drop(labels=["attrelid", "attnum"], axis=1, inplace=True)
        time_pg_index.sort_index(axis=0, inplace=True)

    for i, _ in enumerate(indkeys_cols):
        key = f"indkey_attvarying_{i}"
        base = f"indkey_attlen_{i}"
        time_pg_index[key] = (time_pg_index[base] == -1).astype(int)

    del pg_attribute
    return time_pg_index


def process_time_pg_trigger(time_pg_trigger):
    PG_TRIGGER_SCHEMA = [
        "oid",
        "tgrelid",
        #"tgparentid",
        "tgname",
        "tgfoid",
        "tgtype",
        "tgenabled",
        #"tgisinternal",
        "tgconstrrelid",
        "tgconstrindid",
        "tgconstraint",
        "tgdeferrable",
        #"tginitdeferred",
        #"tgnargs",
        #"tgattr",
        #"tgargs",
        #"tgqual",
        #"tgoldtable",
        #"tgnewtable",
    ]

    cols_remove = [col for col in time_pg_trigger.columns if col not in PG_TRIGGER_SCHEMA]
    time_pg_trigger["unix_timestamp"] = time_pg_trigger.time.astype(float) / 1e6
    time_pg_trigger.drop(labels=cols_remove, axis=1, inplace=True, errors='ignore')
    time_pg_trigger.rename(columns={"oid":"pg_trigger_oid"}, inplace=True)
    return time_pg_trigger


def process_time_pg_constraint(time_pg_constraint):
    PG_CONSTRAINT_SCHEMA = [
        "oid",
        "connname",
        #"connnamespace",
        "contype",
        "condeferrable",
        "condeferred",
        #"convalidated",
        "conrelid",
        "contypid",
        "conindid",
        #"conparentid",
        "confrelid",
        "confupdtype",
        "confdeltype",
        #"confmatchtype",
        #"conislocal",
        #"coninhcount",
        #"connoinherit",
        "conkey",
        "confkey",
        "conpfeqop",
        "conppeqop",
        "conffeqop",
        "conexclop",
        #"conbin",
    ]

    cols_remove = [col for col in time_pg_constraint.columns if col not in PG_CONSTRAINT_SCHEMA]
    time_pg_constraint['unix_timestamp'] = time_pg_constraint.time.astype(float) / 1e6
    time_pg_constraint.drop(labels=cols_remove, axis=1, inplace=True, errors='ignore')
    time_pg_constraint.rename(columns={"oid":"pg_constraint_oid"}, inplace=True)
    return time_pg_constraint


def build_time_trigger_metadata(time_pg_trigger, time_pg_constraint):
    time_pg_trigger.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    time_pg_trigger.sort_index(axis=0, inplace=True)
    if len(time_pg_trigger) == 0:
        # Just return an empty dataframe I guess.
        return time_pg_trigger

    time_pg_constraint.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    time_pg_constraint.sort_index(axis=0, inplace=True)

    data = pd.merge_asof(time_pg_trigger, time_pg_constraint, left_by=["tgconstraint"], right_by=["pg_constraint_oid"], left_index=True, right_index=True, allow_exact_matches=True)
    data.drop(data[data.pg_constraint_oid.isna()].index, inplace=True)
    data.drop(labels=["pg_constraint_oid", "tgconstraint"], axis=1, inplace=True)
    data["pg_trigger_oid"] = data.pg_trigger_oid.astype(np.int64)
    data.reset_index(drop=False, inplace=True)

    data.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    data.sort_index(axis=0, inplace=True)
    return data


def construct_all_time_metadatas(root):
    time_pg_trigger = process_time_pg_trigger(pd.read_csv(root / "pg_trigger.csv"))
    time_pg_constraint = process_time_pg_constraint(pd.read_csv(root / "pg_constraint.csv"))
    time_trigger_metadata = build_time_trigger_metadata(time_pg_trigger, time_pg_constraint)

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
    return time_index_metadata, time_pg_settings, time_pg_stats, time_trigger_metadata


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
        data = pd.read_feather(root / f"{name}.feather")
        data["data_identifier"] = data.index + 1
        data["unix_timestamp"] = postgres_julian_to_unix(data.statement_timestamp)
        data.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
        data.sort_index(axis=0, inplace=True)

        # Read in the pg_class information and extract only tabular information.
        time_tables, _ = process_time_pg_class(pd.read_csv(root / "pg_class.csv"))
        time_tables.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
        time_tables.sort_index(axis=0, inplace=True)

        # Merge the pg_class table against ModifYTableInsert.
        time_data = pd.merge_asof(data, time_tables, left_index=True, right_index=True, left_by=["ModifyTable_target_oid"], right_by=["oid"], allow_exact_matches=True)
        time_data.drop(time_data[time_data.oid.isna()].index, inplace=True)
        time_data.reset_index(drop=False, inplace=True)
        return time_data


    def process_aqt(self, root, time_trigger_metadata):
        logger.info("Processing AfterQueryTrigger")
        data = pd.read_feather(root / "AfterQueryTrigger.feather")
        data["data_identifier"] = data.index + 1
        data["unix_timestamp"] = postgres_julian_to_unix(data.statement_timestamp)
        data.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
        data.sort_index(axis=0, inplace=True)

        time_data = pd.merge_asof(data, time_trigger_metadata, left_index=True, right_index=True, left_by=["AfterQueryTrigger_tgoid"], right_by=["pg_trigger_oid"], allow_exact_matches=True)
        time_data.drop(time_data[time_data.pg_trigger_oid.isna()].index, inplace=True)
        time_data.drop(labels=["AfterQueryTrigger_tgoid"], axis=1, inplace=True)
        time_data.reset_index(drop=False, inplace=True)
        return time_data


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
                time_index_metadata, time_pg_settings, time_pg_stats, time_trigger_metadata = construct_all_time_metadatas(experiment_root / bench_name)

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
                    elif ou == OperatingUnit.AfterQueryTrigger:
                        data = self.process_aqt(experiment_root / bench_name, time_trigger_metadata)
                        logger.info("Writing out %s data", f.stem)
                        data.reset_index(drop=True, inplace=True)
                        data.to_feather(f"{merge_data_dir}/{f.stem}.feather")
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
                time_trigger_metadata.reset_index(drop=True, inplace=True)

                time_index_metadata.to_feather(f"{merge_data_dir}/idx_index_metadata.feather")
                time_pg_settings.to_feather(f"{merge_data_dir}/idx_pg_settings.feather")
                time_pg_stats.to_feather(f"{merge_data_dir}/idx_pg_stats.feather")
                time_trigger_metadata.to_feather(f"{merge_data_dir}/idx_trigger_metadata.feather")

if __name__ == "__main__":
    StateMergeCLI.run()
