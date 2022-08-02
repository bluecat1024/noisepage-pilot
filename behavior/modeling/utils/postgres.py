import copy
import re
import psycopg
from psycopg.rows import dict_row
from distutils import util
from pathlib import Path
from enum import Enum, auto, unique
import pandas as pd
import numpy as np
from behavior.modeling import featurize
from behavior.datagen.pg_collector_utils import SettingType, _time_unit_to_ms, _parse_field, KNOBS
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def prepare_pg_inference_state(conn):
    metadata = {}
    def extract_idx_key(results):
        mapping = {}
        for result in results:
            idxname = result["relname"]
            if idxname not in metadata["pg_class"]:
                continue

            metadata["pg_class"][result["relname"]]["est_key_size"] = result["est_key_size"]

            # Add to the indexes list of the table.
            tbl = result["tblname"]
            metadata["pg_class"][tbl]["indexes"].append(result["relname"])
            mapping[result["indexrelid"]] = result["relname"]

        metadata["pg_index_lookup"] = mapping

    def extract_tuple_size(results):
        for result in results:
            relname = result["relname"]
            if relname not in metadata["pg_class"]:
                continue

            metadata["pg_class"][relname]["est_tuple_size"] = result["est_tuple_size"]

    def pg_class_transform(x):
        # Install table entry under both relname and oid lookup.
        mapping = {}
        for elem in x:
            if "reloptions" in elem and elem["reloptions"] is not None:
                # Supplement the pg_class entry with reloptions.
                for reloption in elem["reloptions"]:
                    for key, value in re.findall(r'(\w+)=(\w*)', reloption):
                        if key == "fillfactor":
                            # Fix fillfactor options.
                            value = float(value) / 100.0
                        elem[key] = value

            elem["indexes"] = []
            mapping[elem["relname"]] = elem

        # Store an indirection layer from OID -> name.
        if "pg_class_lookup" not in metadata:
            metadata["pg_class_lookup"] = {}
        for elem in x:
            metadata["pg_class_lookup"][elem["oid"]] = elem["relname"]
        return mapping

    queries = [
        ("pg_class", """
            SELECT  c.oid,
                    c.relname,
                    c.relpages,
                    c.reltuples,
                    c.relkind,
                    c.reloptions,
                    s.n_dead_tup,
                    s.n_live_tup + s.n_dead_tup as "est_valid_slots",
                    pg_relation_size(c.relname::regclass, 'main'::text) as "est_relation_size"
                from pg_class c
                JOIN pg_namespace n ON c.relnamespace = n.oid and n.nspname = 'public'
                JOIN pg_stat_user_tables s ON s.relname = c.relname
            """, pg_class_transform),

        ("pg_class", """
            SELECT  c.oid,
                    c.relname,
                    c.relpages,
                    c.reltuples,
                    c.relkind,
                    t.n_live_tup + t.n_dead_tup as "est_valid_slots",
                    pg_relation_size(c.relname::regclass, 'main'::text) as "est_relation_size"
                from pg_class c
                JOIN pg_namespace n ON c.relnamespace = n.oid and n.nspname = 'public'
                JOIN pg_stat_user_indexes s ON s.indexrelname = c.relname
                JOIN pg_stat_user_tables t ON t.relname = s.relname
            """, pg_class_transform),

        ("pg_trigger", """
            SELECT t.oid as "pg_trigger_oid", t.tgfoid, c.contype, c.confrelid, c.confupdtype, c.confdeltype, c.conkey, c.confkey, c.conpfeqop
            FROM pg_trigger t, pg_constraint c
            JOIN pg_namespace n ON c.connamespace = n.oid and n.nspname = 'public'
            WHERE t.tgconstraint = c.oid
            """, lambda a: {x["pg_trigger_oid"]: x for x in a}),

        ("pg_attribute", """
            SELECT * FROM pg_attribute
              JOIN pg_class c ON pg_attribute.attrelid = c.oid
              JOIN pg_namespace n ON c.relnamespace = n.oid and n.nspname = 'public'
            """, lambda a: {(x["attrelid"], x["attnum"]): x for x in a}),

        (None, """
        SELECT i.indexrelid, i.indrelid, idx.relname, tbl.relname as "tblname", SUM(s.avg_width) as "est_key_size"
          FROM pg_index i
          JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
          JOIN pg_class tbl ON i.indrelid = tbl.oid
          JOIN pg_class idx ON i.indexrelid = idx.oid
          JOIN pg_stats s ON s.tablename = tbl.relname AND s.attname = a.attname
          JOIN pg_namespace n ON tbl.relnamespace = n.oid and n.nspname = 'public'
      GROUP BY i.indexrelid, i.indrelid, idx.relname, tblname;
        """, extract_idx_key),

        (None, """
        SELECT c.relname, SUM(s.avg_width) as "est_tuple_size"
          FROM pg_class c
          JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum >= 1 AND a.attnum <= c.relnatts
          JOIN pg_stats s ON s.tablename = c.relname AND s.attname = a.attname
          JOIN pg_namespace n ON c.relnamespace = n.oid and n.nspname = 'public'
         WHERE c.relkind = 'r'
      GROUP BY c.relname;
        """, extract_tuple_size),
    ]

    with conn.cursor(row_factory=dict_row) as cursor:
        # Extract all the relevant settings that we care about.
        cursor.execute("SHOW ALL;")
        pg_settings = {}
        for record in cursor:
            setting_name = record["name"]
            if setting_name in KNOBS:
                # Map a pg_setting name to the setting value.
                setting_type = KNOBS[setting_name]
                setting_str = record["setting"]
                pg_settings[setting_name] = _parse_field(setting_type, setting_str)
        metadata['pg_settings'] = pg_settings

        for query in queries:
            cursor.execute(query[1])
            records = [record for record in cursor]
            transform = query[2](records)
            if query[0] is not None:
                if query[0] in metadata:
                    metadata[query[0]].update(transform)
                else:
                    metadata[query[0]] = transform

    return metadata


def prepare_augmentation_data(sliced_metadata, conn):
    assert len(sliced_metadata) > 0

    dfs = {}

    # Add pg_settings scraped.
    settings = sliced_metadata[0]["pg_settings"]
    settings["unix_timestamp"] = 0
    dfs["pg_settings"] = pd.DataFrame([settings])
    with conn.cursor(row_factory=dict_row) as cursor:
        implants = ["pg_class", "pg_index"]
        for implant in implants:
            result = cursor.execute(f"SELECT * FROM {implant}")
            records = [record for record in result]
            all_records = []
            for i, metadata in enumerate(sliced_metadata):
                for record in records:
                    if "relname" in record and record["relname"] in metadata[implant]:
                        relname = record["relname"]
                        record["reltuples"] = metadata[implant][relname]["reltuples"]
                        record["relpages"] = metadata[implant][relname]["relpages"]
                        record["time"] = i * 1.0 * 1e6
                        all_records.append(copy.deepcopy(record))
                    elif "indexrelid" in record and record["indexrelid"] in metadata["pg_index_lookup"]:
                        idxname = metadata["pg_index_lookup"][record["indexrelid"]]
                        if idxname in metadata["pg_class"]:
                            record["time"] = i * 1.0 * 1e6
                            all_records.append(copy.deepcopy(record))
            dfs[implant] = pd.DataFrame(all_records)

        queries = [
            ("pg_attribute", "SELECT * FROM pg_attribute"),
            ("pg_stats", "SELECT * FROM pg_stats WHERE schemaname = 'public'"),
        ]

        for output, query in queries:
            result = cursor.execute(query)
            records = []
            for record in result:
                record["time"] = 0.0
                records.append(record)
            dfs[output] = pd.DataFrame(records)

    return dfs
