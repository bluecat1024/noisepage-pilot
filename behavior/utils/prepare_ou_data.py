import pandas as pd
import numpy as np
from behavior import TARGET_COLUMNS
from behavior.model_ous.process import DIFF_SCHEMA_METADATA


# Any feature that ends with any keyword in BLOCKED_FEATURES is dropped
# from the input schema. These features are intentionally dropped since
# we don't want feature selection/model to try and learn from these.
TRAILING_BLOCKED_FEATURES = DIFF_SCHEMA_METADATA + [
    "plan_type",
    "cpu_id",
    "relid",
    "indexid",
    "unix_timestamp",
    "relname",
    "relkind",
]


def purify_index_input_data(df):
    # This logic is used to replace all the indkey n_distinct_ data (make it positive).
    distinct_keys = [col for col in df.columns if "indkey_n_distinct_" in col]
    filter_check = lambda x: x >= 0
    other_data = lambda x: (-x) * df["table_reltuples"]
    for distinct_key in distinct_keys:
        # Switch the negative distinct to a positive based on num tuples in table
        df[distinct_key] = df[distinct_key].where(filter_check, other=other_data)
        # make sure there are no longer any negative numbers in distinct...
        assert (df[distinct_key] < 0).sum() == 0

    # TODO(wz2): Here we try to encode summary statistics as much as possible.
    slot = [
        "indkey_attlen_",
        "indkey_atttypmod_",
        "indkey_attvarying_",
        "indkey_avg_width_",

        # TODO(wz2): Null Fraction is dropped since there's no good way of representing this.
        # Models probably also don't need this. Similar reasoning for correlation.
        "indkey_n_distinct_",
        "indkey_null_frac_",
        "indkey_correlation_",
    ]
    blocked = ([col for col in df.columns for prefix in slot if prefix in col])
    if len(blocked) > 0:
        varying_keys = [col for col in df.columns if "indkey_attvarying_" in col]
        attlen_keys = [col for col in df.columns if "indkey_attlen_" in col]
        atttypmod_keys = [col for col in df.columns if "indkey_attypmod_" in col]
        avg_width_keys = [col for col in df.columns if "indkey_avg_width_" in col]

        df["indkey_cum_attvarying"] = 0
        df["indkey_cum_attfixed"] = 0
        df["indkey_cum_attlen"] = 0
        df["indkey_cum_avg_width"] = 0
        for col in varying_keys:
            df["indkey_cum_attvarying"] += (df[col].fillna(0))
        for col in avg_width_keys:
            df["indkey_cum_avg_width"] += (df[col].fillna(0))
        for col in attlen_keys:
            df["indkey_cum_attfixed"] += ((df[col].fillna(0)) != -1).astype(int)
        filter_check = lambda x: x >= 0
        for col in attlen_keys:
            df["indkey_cum_attlen"] += (df[col].fillna(0)).where(filter_check, 0)
        for col in atttypmod_keys:
            df["indkey_cum_attlen"] += (df[col].fillna(0)).where(filter_check, 0)
        df.drop(columns=blocked, inplace=True, errors='raise')
    return df


def clean_input_data(df, settings, stats, is_train):
    """
    Function prepares input data for an OU.

    Parameters
    ----------
    df : DataFrame
        Input (train/test) data for an operating unit.
    settings : DataFrame
        Dataframe for corresponding settings data with timestamps and identifiers.
    stats : DataFrame
        Dataframe for corresponding stats data with timestamps and identifiers.
    is_train : bool
        Whether dataframe is used for training. Columns are not dropped for non training dataframes.
    """
    # join against the settings.
    if "pg_settings_identifier" in df.columns:
        df.set_index(keys=["pg_settings_identifier"], drop=True, append=False, inplace=True)
        settings = settings.set_index(keys=["pg_settings_identifier"], drop=True, append=False)
        df = df.join(settings, how="inner")
        df.reset_index(drop=True, inplace=True)

    # join against the stats identifier...this is a bit sadness but ok!
    stats_ids = [key for key in df.columns if "indkey_statsid_" in key]
    if len(stats_ids) != 0:
        stats = stats.set_index(keys=["pg_stats_identifier"], drop=True, append=False)
        stats.drop(labels=["time", "tablename", "attname"], axis=1, inplace=True)

        for idx, stats_id_key in enumerate(stats_ids):
            df.set_index(keys=[stats_id_key], drop=True, append=False, inplace=True)
            df = df.join(stats, how="left")
            df.reset_index(drop=True, inplace=True)

            # rename the columns that were just inserted.
            remapper = {column:f"indkey_{column}_{idx}" for column in stats.columns}
            df.rename(columns=remapper, inplace=True)

    # Remove all features that are blocked.
    blocked = [col for col in df.columns for block in TRAILING_BLOCKED_FEATURES if col.endswith(block)]

    # TODO(wz2): Do we really want to drop all of these?
    # Drop all these additional metadata or useless target columns
    blocked.extend([
        "unix_timestamp",
        "time",

        "relname",
        "relnatts", # redundant since just # of key+include in index attributes
        "table_relname",
        "table_relkind",
        # indnatts - # of key+include in index attributes
        # indnkeyatts - # of key in index attributes
        # table_relnatts -- # of attributes in the table
    ])

    slot = [
        "indkey_slot_", # index of the pg_attribute
        "indkey_attname_", # name of the attribute
        "indkey_most_common_vals_", # most common vals
        "indkey_most_common_freqs_", # most common freqs
        "indkey_histogram_bounds_", # histogram bounds
    ]
    blocked.extend([col for col in df.columns for prefix in slot if prefix in col])

    # Eliminate all columns that end with OID
    blocked.extend([col for col in df.columns if col.endswith("oid")])

    # Clean and perform any relevant operations on the input data.
    df = purify_index_input_data(df)

    # This is an OU-specific dataframe operation. Why? Well because we want the num_outer_loops
    # to handle the query-level "nested" behavior.
    if "IndexScan_num_outer_loops" in df.columns:
        div_cols = TARGET_COLUMNS + ["IndexScan_num_iterator_used", "IndexScan_num_heap_fetches"]
        div_cols = [col for col in div_cols if col in df]
        if len(div_cols) > 0:
            df[div_cols] = df[div_cols].div(df.IndexScan_num_outer_loops, axis=0)
            df.drop(columns=["IndexScan_num_outer_loops"], inplace=True, errors='raise')

    # TODO(wz2): I think we should drop state variables other than below? Why?
    # Because these are transitory and sort of capture the "end" of execution state
    # which will be fundamentally different to when we do inference.
    state_include = ["NumScanKeys", "NumOrderByKeys", "NumRuntimeKeys"]
    state_include = [col for col in df.columns for inc in state_include if inc in col]
    blocked.extend([col for col in df.columns if "ScanState" in col and col not in state_include])

    if is_train:
        # Only drop the columns in train. Evaluate should probably have all the columns for analysis.
        df.drop(blocked, axis=1, inplace=True, errors='ignore')

    df.fillna(0.0, inplace=True)

    # Massage these column datatypes.
    if "indisunique" in df.columns:
        df["indisunique"] = (df.indisunique == "t").astype(int)

    if "indisprimary" in df.columns:
        df["indisprimary"] = (df.indisprimary == "t").astype(int)

    if "indisexclusion" in df.columns:
        df["indisexclusion"] = (df.indisexclusion == "t").astype(int)

    if "indimmediate" in df.columns:
        df["indimmediate"] = (df.indimmediate == "t").astype(int)

    # Convert all bool type columns into an integer column.
    for i, dtype in enumerate(df.dtypes):
        if dtype == "bool":
            df[df.columns[i]] = df[df.columns[i]].astype(int)

    # Sort the DataFrame by column for uniform downstream outputs.
    df.sort_index(axis=1, inplace=True)

    # Featurewiz expects the TARGET_COLUMNS to be at the end for some reason.
    df = df[[c for c in df if c not in TARGET_COLUMNS] + [t for t in TARGET_COLUMNS if t in df]]
    return df


def load_input_data(logger, path, source_map, is_train):
    if logger:
        logger.info("Prepping input data for file: %s", path)

    settings = pd.read_feather(path.parent / "idx_pg_settings.feather")
    stats = pd.read_feather(path.parent / "idx_pg_stats.feather")

    df = clean_input_data(pd.read_feather(path), settings, stats, is_train)

    file_idx = len(source_map) + 1
    source_map[file_idx] = path.name
    df["source_file"] = file_idx
    return df
