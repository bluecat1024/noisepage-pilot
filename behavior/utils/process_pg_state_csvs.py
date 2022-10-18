import pandas as pd


def postgres_julian_to_unix(df):
    # Formula to convert postgres JULIAN to unix timestamp.
    return (df / float(1e6)) + 86400 * (2451545 - 2440588)


PG_STATS_SCHEMA = [
    "tablename",
    "attname",
    # "inherited",
    "null_frac",
    "avg_width",
    "n_distinct",
    # "most_common_vals",
    # "most_common_freqs",
    # "histogram_bounds",
    "correlation",
    # "most_common_elems",
    # "most_common_elem_freqs",
    # "elem_count_histogram",
]


def process_time_pg_stats(time_pg_stats):
    cols_remove = [col for col in time_pg_stats.columns if col not in PG_STATS_SCHEMA]

    public_mask = (time_pg_stats.schemaname != 'public')
    time_pg_stats.drop(time_pg_stats[public_mask].index, axis=0, inplace=True)
    time_pg_stats["unix_timestamp"] = time_pg_stats.time.astype(float) / 1e6

    time_pg_stats.drop(labels=cols_remove, axis=1, inplace=True, errors='ignore')
    time_pg_stats.reset_index(drop=True, inplace=True)
    time_pg_stats["pg_stats_identifier"] = time_pg_stats.index
    return time_pg_stats


PG_CLASS_SCHEMA = [
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


PG_CLASS_INDEX_SCHEMA = [
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
    # "relallvisible",
    # "reltoastrelid",
    # "relhasindex",
    # "relisshared",
    # "relpersistence",
    # "relkind",
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


def process_time_pg_class(time_pg_class):
    cols_remove = [col for col in time_pg_class.columns if col not in PG_CLASS_SCHEMA]
    time_pg_class["unix_timestamp"] = time_pg_class.time.astype(float) / 1e6
    time_pg_class = time_pg_class.drop(labels=cols_remove, axis=1, errors='ignore')
    time_pg_class.reset_index(drop=True, inplace=True)

    tables = time_pg_class[(time_pg_class.relkind == 'r') | (time_pg_class.relkind == 't')]
    indexes = time_pg_class[time_pg_class.relkind == 'i']
    indexes = indexes[["unix_timestamp"] + PG_CLASS_INDEX_SCHEMA].copy()

    tables.reset_index(drop=True, inplace=True)
    indexes.reset_index(drop=True, inplace=True)
    return tables, indexes


def merge_modifytable_data(name=None, root=None, data=None, pg_class=None, processed_time_tables=None):
    if name is not None:
        assert data is None and pg_class is None
        data = pd.read_feather(root / f"{name}.feather")
        data["data_identifier"] = data.index + 1
        data["unix_timestamp"] = postgres_julian_to_unix(data.statement_timestamp)
        data.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
        data.sort_index(axis=0, inplace=True)

        # Read in the pg_class information and extract only tabular information.
        pg_class = pd.read_csv(root / "pg_class.csv")
        time_tables, _ = process_time_pg_class(pg_class)
        time_tables.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
        time_tables.sort_index(axis=0, inplace=True)
    else:
        data.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
        data.sort_index(axis=0, inplace=True)
        time_tables = processed_time_tables

    # Merge the pg_class table against ModifyTableInsert/Update.
    time_data = pd.merge_asof(data, time_tables, left_index=True, right_index=True, left_by=["ModifyTable_target_oid"], right_by=["oid"], allow_exact_matches=True)
    time_data.reset_index(drop=False, inplace=True)
    time_data.drop(time_data[time_data.oid.isna()].index, inplace=True)
    time_data.reset_index(drop=True, inplace=True)
    return time_data


PG_ATTRIBUTE_SCHEMA = [
    "attrelid",
    "attname",
    # "atttypid",
    # "attstattarget",
    "attlen",
    "attnum",
    # "attndims",
    # "attcacheoff",
    "atttypmod",
    # "attbyval",
    # "attalign",
    # "attstorage",
    # "attcompression",
    # "attnotnull",
    # "atthasdef",
    # "atthasmissing",
    # "attidentity",
    # "attgenerated",
    # "attisdropped",
    # "attislocal",
    # "attinhcount",
    # "attcollation",
    # "attacl",
    # "attoptions",
    # "attfdwoptions",
    # "attmissingval",
]


def process_time_pg_attribute(time_pg_attribute):
    cols_remove = [col for col in time_pg_attribute.columns if col not in PG_ATTRIBUTE_SCHEMA]
    time_pg_attribute["unix_timestamp"] = time_pg_attribute.time.astype(float) / 1e6
    time_pg_attribute.drop(labels=cols_remove, axis=1, inplace=True, errors='ignore')
    return time_pg_attribute


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


def process_time_pg_index(time_pg_index):
    cols_remove = [col for col in time_pg_index.columns if col not in PG_INDEX_SCHEMA]
    time_pg_index["unix_timestamp"] = time_pg_index.time.astype(float) / 1e6
    time_pg_index.drop(labels=cols_remove, axis=1, inplace=True, errors='ignore')
    return time_pg_index


def build_time_index_metadata(pg_index, tables, cls_indexes, pg_attribute):
    # First order of attack is to join cls_indexes against pg_index.
    cls_indexes.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    cls_indexes.sort_index(axis=0, inplace=True)
    pg_index.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    pg_index.sort_index(axis=0, inplace=True)

    time_pg_index = pd.merge_asof(cls_indexes, pg_index, left_index=True, right_index=True, left_by=["oid"], right_by=["indexrelid"], allow_exact_matches=True)
    time_pg_index.reset_index(drop=False, inplace=True)
    time_pg_index.drop(time_pg_index[time_pg_index.indexrelid.isna()].index, inplace=True)
    time_pg_index.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)

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
    time_pg_index.reset_index(drop=False, inplace=True)
    time_pg_index.drop(time_pg_index[time_pg_index.table_oid.isna()].index, inplace=True)
    time_pg_index.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
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


def process_time_pg_settings(time_pg_settings):
    time_pg_settings["unix_timestamp"] = time_pg_settings.time.astype(float) / 1e6
    time_pg_settings.reset_index(drop=True, inplace=True)
    time_pg_settings["pg_settings_identifier"] = time_pg_settings.index
    return time_pg_settings
