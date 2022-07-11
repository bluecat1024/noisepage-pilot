import gc
import re
import json
from tqdm import tqdm
from pathlib import Path
from enum import Enum, auto, unique
import pandas as pd
import numpy as np
from behavior import TARGET_COLUMNS
from behavior.plans import DIFF_SCHEMA_METADATA
from behavior.modeling import featurize


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
        "indkey_n_distinct_",
        "indkey_avg_width_",

        # TODO(wz2): Null Fraction is dropped since there's no good way of representing this.
        # Models probably also don't need this. Similar reasoning for correlation.
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
        "cpu_cycles",
        "instructions",
        "cache_references",
        "cache_misses",
        "ref_cpu_cycles",
        "network_bytes_read",
        "network_bytes_written",
        "disk_bytes_read",
        "disk_bytes_written",
        "memory_bytes",
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


def prepare_inference_query_stream(dir_data):
    # If any of these query content fragments are found within the query_text,
    # those queries are omitted. This helps remove any queries that we don't
    # care about from the analysis.
    QUERY_CONTENT_BLOCK = [
        "pg_",
        "version()",
        "current_schema()",
    ]

    # Load all the "true" query stats associated with runtime.
    query_id_index = ["query_id", "db_id", "pid"]
    pg_qss_stats = pd.read_csv(dir_data / "pg_qss_stats.csv")
    assert pg_qss_stats.shape[0] > 0
    query_stats = pg_qss_stats[pg_qss_stats.plan_node_id == -1]
    query_stats.rename(columns={"counter0": "total_elapsed_us"}, inplace=True)
    query_stats.drop(columns=[f"counter{i}" for i in range(1, 10)] + ["plan_node_id"], inplace=True, errors='raise')
    query_stats["params"] = query_stats.params.fillna('')
    del pg_qss_stats
    gc.collect()

    # Process pg_qss_plans to extract the query_text.
    pg_qss_plans = pd.read_csv(dir_data / "pg_qss_plans.csv")
    pg_qss_plans["id"] = pg_qss_plans.index
    pg_qss_plans["query_text"] = ""
    for plan in pg_qss_plans.itertuples():
        feature = json.loads(plan.features)
        query_text = feature[0]["query_text"].lower()

        blocked = False
        for query in QUERY_CONTENT_BLOCK:
            if query_text is not None and query in query_text:
                query_text = None
                blocked = True
        pg_qss_plans.at[plan.Index, "query_text"] = query_text
    pg_qss_plans.drop(labels=["features"], axis=1, inplace=True)
    pg_qss_plans.drop(pg_qss_plans[pg_qss_plans.query_text.isna()].index, inplace=True)
    gc.collect()

    # Combine the query stats to the query text that generated it.
    pg_qss_plans.reset_index(drop=True, inplace=True)
    pg_qss_plans.set_index(keys=["statement_timestamp"], drop=True, append=False, inplace=True)
    pg_qss_plans.sort_index(axis=0, inplace=True)

    query_stats.set_index(keys=["statement_timestamp"], drop=True, append=False, inplace=True)
    query_stats.sort_index(axis=0, inplace=True)

    query_stats = pd.merge_asof(query_stats, pg_qss_plans, left_by=query_id_index, right_by=query_id_index, left_index=True, right_index=True, allow_exact_matches=True)
    query_stats.reset_index(drop=False, inplace=True)
    query_stats.drop(query_stats[query_stats.query_text.isna()].index, inplace=True)
    del pg_qss_plans
    gc.collect()

    # Parametrize the query. (not done with apply due to memory constraints).
    for query in tqdm(query_stats.itertuples(), total=query_stats.shape[0]):
        matches = re.findall(r'(\$\w+) = (\'(?:[^\']*(?:\'\')?[^\']*)*\')', query.params)
        query_text = query.query_text
        if len(matches) > 0:
            parts = []
            for match in matches:
                start = query_text.find(match[0])
                parts.append(query_text[:start])
                parts.append(match[1])
                query_text = query_text[start+len(match[0]):]
            parts.append(query_text)
            query_text = "".join(parts)
        query_stats.at[query.Index, "query_text"] = query_text
    query_stats.drop(labels=["params", "id", "generation"], axis=1, inplace=True)

    # Sort by statement_timestamp so we are processing the query stream in serial.
    query_stats.set_index(keys=["statement_timestamp"], drop=True, append=False, inplace=True)
    query_stats.sort_index(axis=0, inplace=True)
    # We don't actually want the statement_timestamp since that is "real" time but we do want the queries
    # in the correct execution order.
    query_stats["order"] = np.arange(len(query_stats))
    query_stats.reset_index(drop=True, inplace=True)
    return query_stats
