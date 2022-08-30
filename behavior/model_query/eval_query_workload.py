import random
import pglast
from tqdm import tqdm
import logging
import glob
import shutil
import gc
import re
import psycopg
import copy
from psycopg.rows import dict_row
from distutils import util
import json
from datetime import datetime
import numpy as np
import itertools
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from plumbum import cli

from behavior import OperatingUnit, Targets, BENCHDB_TO_TABLES
from behavior.utils.evaluate_ou import evaluate_ou_model
from behavior.utils.prepare_ou_data import purify_index_input_data
from behavior.model_query.utils.postgres import prepare_augmentation_data, prepare_pg_inference_state
from behavior.model_query.utils.prepare_data import prepare_inference_query_stream
from behavior.model_query.utils.postgres_model_plan import generate_vacuum_partition, generate_query_ous, estimate_query_modifications
from behavior.model_workload.utils import compute_frame
from behavior.utils.process_pg_state_csvs import (
    process_time_pg_stats,
    process_time_pg_attribute,
    process_time_pg_index,
    process_time_pg_class,
    merge_modifytable_data,
    build_time_index_metadata
)
from behavior.model_workload.model import WorkloadModel
import torch

logger = logging.getLogger(__name__)


def augment_ous(scratch_it, sliced_metadata, conn):
    if ((scratch_it / "augment_query_ous")).exists():
        return

    # Prepare all the augmented catalog data in timestamp order.
    aug_dfs = prepare_augmentation_data(sliced_metadata, conn)
    process_tables, process_idxs = process_time_pg_class(aug_dfs["pg_class"])
    process_pg_index = process_time_pg_index(aug_dfs["pg_index"])
    process_pg_attribute = process_time_pg_attribute(aug_dfs["pg_attribute"])
    time_pg_index = build_time_index_metadata(process_pg_index, process_tables.copy(deep=True), process_idxs, process_pg_attribute)

    time_pg_stats = process_time_pg_stats(aug_dfs["pg_stats"])
    time_pg_stats.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    time_pg_stats.sort_index(axis=0, inplace=True)

    pg_settings = aug_dfs["pg_settings"]
    pg_settings["unix_timestamp"] = pg_settings.unix_timestamp.astype(np.float64)
    pg_settings.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    pg_settings.sort_index(axis=0, inplace=True)

    index_augment = [OperatingUnit.IndexScan, OperatingUnit.IndexOnlyScan, OperatingUnit.ModifyTableIndexInsert]
    for augment in index_augment:
        column = {
            OperatingUnit.IndexOnlyScan: "IndexOnlyScan_indexid",
            OperatingUnit.IndexScan: "IndexScan_indexid",
            OperatingUnit.ModifyTableIndexInsert: "ModifyTableIndexInsert_indexid"
        }[augment]

        files = sorted(glob.glob(f"{scratch_it}/{augment.name}.feather.*"))
        for target_file in files:
            if target_file.startswith("AUG"):
                continue

            logger.info("[AUGMENT] Input %s", target_file)
            data = pd.read_feather(target_file)
            data.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
            data.sort_index(axis=0, inplace=True)

            data = pd.merge_asof(data, pg_settings, left_index=True, right_index=True, allow_exact_matches=True)
            data = pd.merge_asof(data, time_pg_index, left_index=True, right_index=True, left_by=[column], right_by=["indexrelid"], allow_exact_matches=True)
            assert data.indexrelid.isna().sum() == 0

            indkey_atts = [key for key in data.columns if "indkey_attname_" in key]
            for idx, indkey_att in enumerate(indkey_atts):
                left_by = ["table_relname", indkey_att]
                right_by = ["tablename", "attname"]
                data = pd.merge_asof(data, time_pg_stats, left_index=True, right_index=True, left_by=left_by, right_by=right_by, allow_exact_matches=True)

                # Rename the key and drop the other useless columns.
                data.drop(labels=["tablename", "attname"], axis=1, inplace=True)
                remapper = {column:f"indkey_{column}_{idx}" for column in time_pg_stats.columns}
                data.rename(columns=remapper, inplace=True)

            data = purify_index_input_data(data)
            data.reset_index(drop=True, inplace=True)
            data.to_feather(scratch_it / f"AUG_{augment.name}.feather{Path(target_file).suffix}")

    mt_augment = [OperatingUnit.ModifyTableInsert, OperatingUnit.ModifyTableUpdate]
    process_tables.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    process_tables.sort_index(axis=0, inplace=True)
    for augment in mt_augment:
        files = sorted(glob.glob(f"{scratch_it}/{augment.name}.feather.*"))
        for target_file in files:
            logger.info("[AUGMENT] Input %s", target_file)
            data = pd.read_feather(target_file)
            data = merge_modifytable_data(data=data, processed_time_tables=process_tables)
            data.to_feather(scratch_it / f"AUG_{augment.name}.feather{Path(target_file).suffix}")

    open(scratch_it / "augment_query_ous", "w").close()


def evaluate_query_plans(eval_batch_size, base_models, scratch_it, queries, output_path):
    # Evaluate all the OUs.
    for ou in OperatingUnit:
        ou_type = ou.name
        key_fn = lambda x: int(x.split(".")[-1])
        files = sorted(glob.glob(f"{scratch_it}/AUG_{ou.name}.feather.*"), key=key_fn)
        if len(files) == 0:
            files = sorted(glob.glob(f"{scratch_it}/{ou.name}.feather.*"), key=key_fn)

        if len(files) == 0:
            # This case is no data for the OU.
            continue

        def logged_read_feather(target):
            logger.info("[^]: [LOAD] Input %s", target)
            return pd.read_feather(target)

        (output_path / ou_type).mkdir(parents=True, exist_ok=True)
        eval_slices = [files[x:x+eval_batch_size] for x in range(0, len(files), eval_batch_size)]
        for i, eval_slice in enumerate(eval_slices):
            df = pd.concat(map(logged_read_feather, eval_slice))
            df.reset_index(drop=True, inplace=True)

            if ou_type not in base_models:
                # If we don't have the model for the particular OU, we just predict 0.
                df["pred_elapsed_us"] = 0
                # Set a bit in [error_missing_model]
                df["error_missing_model"] = 1
            else:
                df = evaluate_ou_model(base_models[ou_type], None, None, eval_df=df, return_df=True, output=False)
                df["error_missing_model"] = 0

                if OperatingUnit[ou_type] == OperatingUnit.IndexOnlyScan or OperatingUnit[ou_type] == OperatingUnit.IndexScan:
                    prefix = "IndexOnlyScan" if OperatingUnit[ou_type] == OperatingUnit.IndexOnlyScan else "IndexScan"
                    df["pred_elapsed_us"] = df.pred_elapsed_us * df[f"{prefix}_num_outer_loops"]

            df.to_feather(output_path / ou_type / f"{ou_type}_evals_{i}.feather")

    # Massage the frames together from disk to combat OOM.
    unified_df = None
    glob_files = []
    for ou in OperatingUnit:
        ou_type = ou.name
        glob_files.extend(glob.glob(f"{output_path}/{ou_type}/{ou_type}_*.feather"))

    assert len(glob_files) > 0
    key = lambda x: int(x.split(".feather")[0].split("_")[-1])
    glob_files = sorted(glob_files, key=key)
    for ou_file in glob_files:
        logger.info("[^]: [MASSAGE] Input %s", ou_file)
        keep_columns = ["query_id", "query_order", "PLOT_TS", "pred_elapsed_us", "error_missing_model"]
        df = pd.read_feather(ou_file, columns=keep_columns)
        if unified_df is None:
            unified_df = df
        else:
            unified_df = pd.concat([unified_df, df], ignore_index=True).groupby(["query_id", "query_order", "PLOT_TS"]).sum()
            unified_df.reset_index(drop=False, inplace=True)
        gc.collect()

    assert unified_df is not None
    unified_df.sort_values(by=["query_order"], inplace=True, ignore_index=True)

    unified_df.set_index(keys=["query_id", "query_order"], inplace=True)
    assert unified_df.index.is_unique

    queries.set_index(keys=["query_id", "query_order"], inplace=True)
    queries = queries.join(unified_df, how="inner")
    queries.reset_index(drop=False, inplace=True)
    assert np.sum(queries.pred_elapsed_us.isna()) == 0
    queries.to_feather(output_path / "query_results.feather")


class Loader():
    def _load_next_chunk(self):
        logger.info("Reading in input chunk: %s", self.files[0])
        self.current_chunk = pd.read_feather(f"{self.files[0]}/chunk.feather")
        self.current_augment_chunk = pd.read_feather(f"{self.files[0]}/chunk_augment.feather")
        self.files = self.files[1:]

        none_slice = self.current_augment_chunk[self.current_augment_chunk.target.isnull()]
        for tbl in self.tables:
            targets = none_slice.query_text.str.contains(tbl)
            self.current_augment_chunk.loc[none_slice[targets].index, "target"] = tbl
        self.current_augment_chunk.set_index(keys=["slot"], inplace=True)

    def __init__(self, dir_data, slice_window, tables):
        super(Loader, self).__init__()
        key_fn = lambda x: int(x.split("_")[-1])
        self.files = sorted(glob.glob(f"{dir_data}/snippets/chunk_*"), key=key_fn)
        self.slice_window = slice_window
        self.tables = tables
        self._load_next_chunk()

    def _get_from_chunk(self, num):
        logger.info("Reading from current chunk: %s", num)
        assert num <= self.current_chunk.shape[0]

        chunk = self.current_chunk.iloc[:num]
        chunk.reset_index(drop=True, inplace=True)
        self.current_chunk = self.current_chunk.iloc[num:]

        # Compute the augmented chunk.
        jchunk = chunk.set_index(keys=["slot"], inplace=False)
        jchunk.drop(columns=[c for c in self.current_augment_chunk], errors='ignore', inplace=True)
        augment_chunk = jchunk.join(self.current_augment_chunk, how="inner")
        augment_chunk.reset_index(drop=False, inplace=True)
        return chunk, augment_chunk

    def get_next_slice(self):
        if self.current_chunk.shape[0] >= self.slice_window:
            return self._get_from_chunk(self.slice_window)

        if self.current_chunk.shape[0] == 0:
            chunk = None
            augment_chunk = None
        else:
            chunk, augment_chunk = self._get_from_chunk(self.current_chunk.shape[0])

        while (chunk is None or chunk.shape[0] < self.slice_window) and len(self.files) > 0:
            # Load the next chunk.
            self._load_next_chunk()

            data_slice = min(self.slice_window - chunk.shape[0], self.current_chunk.shape[0])
            next_chunk, next_augment_chunk = self._get_from_chunk(data_slice)
            chunk = pd.concat([chunk, next_chunk], ignore_index=True)
            augment_chunk = pd.concat([augment_chunk, next_augment_chunk], ignore_index=True)

        return chunk, augment_chunk


def load_models(path):
    model_dict = {}
    for model_path in path.rglob('*.pkl'):
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        model_dict[model.ou_name] = model
    return model_dict


query_cache = {}
def evaluate_query(conn, query):
    if query in query_cache:
        return copy.deepcopy(query_cache[query])
    else:
        matches = re.findall(r'(\$[0-9])', query)
        if len(matches) == 0:
            result = [r for r in conn.execute(f"EXPLAIN (format noisepage) " + query, prepare=False)]
        else:
            conn.execute("DEALLOCATE ALL", prepare=False)
            conn.execute("PREPARE qq AS " + query, prepare=False)

            args = ",".join(["'0'" for i in range(len(matches))])
            result = [r for r in conn.execute(f"EXPLAIN (format noisepage) EXECUTE qq (" + args + ")", prepare=False)]

        result = result[0][0]

        # Extract all the OUs for a given query_text plan.
        features = json.loads(result)[0]
        template_ous = []

        def extract_ou(plan):
            ou = {}
            accum_total_cost = 0.0
            accum_startup_cost = 0.0
            for key in plan:
                value = plan[key]
                if key == "Plans":
                    for p in plan[key]:
                        child_total_cost, child_startup_cost = extract_ou(p)
                        accum_total_cost += child_total_cost
                        accum_startup_cost += child_startup_cost
                    continue
                if isinstance(value, list):
                    # TODO(wz2): For now, we simply featurize a list[] with a numeric length.
                    # This is likely insufficient if the content of the list matters significantly.
                    ou[key + "_len"] = len(value)
                ou[key] = value

            cur_total_cost = ou["total_cost"]
            cur_startup_cost = ou["startup_cost"]
            ou["total_cost"] = max(ou["total_cost"] - accum_total_cost, 0)
            ou["startup_cost"] = max(ou["startup_cost"] - accum_startup_cost, 0)
            ou["query_text"] = query
            template_ous.append(ou)
            return cur_total_cost, cur_startup_cost
        extract_ou(features)
        query_cache[query] = copy.deepcopy(template_ous)
        return template_ous


def augment_single_ou(conn, ous, query, idxoid_table_map, tableoid_map, metadata, percents):
    def get_key(key, other):
        for subst_key in other:
            if key in subst_key:
                return other[subst_key]
        assert False, f"Could not find {key} in {other}"

    def get_plan_rows_matching_plan_id(plan_id):
        for target_ou in ous:
            if target_ou["plan_node_id"] == plan_id:
                for k in target_ou:
                    if "iterator_used" in k:
                        return target_ou[k]
                return get_key("plan_plan_rows", target_ou)
        assert False, f"Could not find plan node with {plan_id}"

    def exist_ou_with_child_plan_id(ou, plan_id):
        for target_ou in ous:
            ou_type = OperatingUnit[target_ou["node_type"]]
            if ou == ou_type and (target_ou["left_child_node_id"] == plan_id or target_ou["right_child_node_id"] == plan_id):
                return target_ou
        return None

    new_ous = []
    def gen_idx_inserts(ou):
        addt_ous = []
        for idxoid in ou["ModifyTable_indexupdates_oids"]:
            idx_name = metadata['pg_class_lookup'][idxoid]
            idx_metadata = metadata['pg_class'][idx_name]
            keys_per_page = 8192 / idx_metadata["est_key_size"]
            if not "inserts" in idx_metadata:
                idx_metadata["inserts"] = 0
            idx_metadata["inserts"] = idx_metadata["inserts"] + 1
            split = (idx_metadata["inserts"] % (keys_per_page / 2)) == 0

            # FIXME(INDEX): Need to use an index learned percentage here.
            addt_ous.append({
                'node_type': OperatingUnit.ModifyTableIndexInsert.name,
                'ModifyTableIndexInsert_indexid': idxoid,
                'plan_node_id': -1,
                # Indicate we split, assume at most one split. Assume split always triggers a relation extend!
                'ModifyTableIndexInsert_num_splits': int(split),
                'ModifyTableIndexInsert_num_extends': int(split),
            })
        return addt_ous

    for ou in ous:
        ou_type = OperatingUnit[ou["node_type"]]
        if ou_type == OperatingUnit.IndexOnlyScan or ou_type == OperatingUnit.IndexScan:
            prefix = "IndexOnlyScan" if ou_type == OperatingUnit.IndexOnlyScan else "IndexScan"
            tbl = idxoid_table_map[ou[prefix + "_indexid"]]
            ou[f"{prefix}_num_outer_loops"] = 1.0

            limit_ou = exist_ou_with_child_plan_id(OperatingUnit.Limit, ou["plan_node_id"])
            if limit_ou is not None:
                # In the case where there is a LIMIT directly above us, only fetch the Limit.
                ou[f"{prefix}_num_iterator_used"] = get_key("plan_rows", limit_ou)
            elif query is None or np.isnan(getattr(query, f"{tbl}_pk_output")):
                ou[f"{prefix}_num_iterator_used"] = get_key("plan_rows", ou)
            else:
                ou[f"{prefix}_num_iterator_used"] = getattr(query, f"{tbl}_pk_output")
            ou[f"{prefix}_num_defrag"] = int(random.uniform(0, 1) <= (percents[tbl]["defrag_percent"] * ou[f"{prefix}_num_iterator_used"]))
        elif ou_type == OperatingUnit.Agg:
            # The number of input rows is the number of rows output by the child.
            ou["Agg_num_input_rows"] = get_plan_rows_matching_plan_id(ou["left_child_node_id"])
        elif ou_type == OperatingUnit.DestReceiverRemote:
            ou["DestReceiverRemote_num_output"] = get_key("plan_rows", ou)
        elif ou_type == OperatingUnit.ModifyTableDelete:
            assert not np.isnan(query.num_modify)
            ou["ModifyTableDelete_num_deletes"] = query.num_modify
        elif ou_type == OperatingUnit.ModifyTableUpdate:
            assert not np.isnan(query.num_modify)
            tbl = tableoid_map[ou["ModifyTable_target_oid"]]
            ou["ModifyTableUpdate_num_updates"] = query.num_modify
            ou["ModifyTableUpdate_num_extends"] = 0
            ou["ModifyTableUpdate_num_hot"] = 0
            for _ in range(int(query.num_modify)):
                if random.uniform(0, 1) <= (percents[tbl]["hot_percent"]):
                    # This is a HOT update...
                    ou["ModifyTableUpdate_num_hot"] = ou["ModifyTableUpdate_num_hot"] + 1
                else:
                    if random.uniform(0, 1) <= (percents[tbl]["extend_percent"]):
                        ou["ModifyTableUpdate_num_extends"] = ou["ModifyTableUpdate_num_extends"] + 1
                    new_ous.extend(gen_idx_inserts(ou))
        elif ou_type == OperatingUnit.ModifyTableInsert:
            tbl = tableoid_map[ou["ModifyTable_target_oid"]]
            ou["ModifyTableInsert_num_extends"] = 0
            if random.uniform(0, 1) <= percents[tbl]["extend_percent"]:
                ou["ModifyTableInsert_num_extends"] = 1
            new_ous.extend(gen_idx_inserts(ou))

            for tgoid in ou["ModifyTable_ar_triggers"]:
                trigger_info = metadata['pg_trigger'][tgoid]
                if trigger_info['contype'] != 'f':
                    # UNIQUE constraints should be handled by indexes.
                    continue

                # 1644 is the hardcoded code for RI_FKey_check_ins.
                assert trigger_info["tgfoid"] == 1644
                frelname = metadata["pg_class_lookup"][trigger_info["confrelid"]]
                tgquery = f"SELECT 1 FROM {frelname} WHERE "

                for i, attnum in enumerate(trigger_info["confkey"]):
                    search = (trigger_info["confrelid"], attnum)
                    att_info = metadata["pg_attribute"][search]
                    if i != 0:
                        tgquery = tgquery + " AND "
                    # TODO(wz2): We currently use a placeholder value. Implication is that for certain cardinalities, this is bust.
                    tgquery = tgquery + f"{att_info['attname']} = '0'";

                # Get the OUs for the trigger query plan and compute the derived OUs.
                tgous = evaluate_query(conn, tgquery)
                new_ous.extend(augment_single_ou(conn, tgous, None, idxoid_table_map, tableoid_map, metadata, percents))

    for ou in ous:
        ou_type = OperatingUnit[ou["node_type"]]
        if ou_type == OperatingUnit.IndexOnlyScan or ou_type == OperatingUnit.IndexScan:
            prefix = "IndexOnlyScan" if ou_type == OperatingUnit.IndexOnlyScan else "IndexScan"
            for other_ou in ous:
                if other_ou["node_type"] == OperatingUnit.NestLoop.name and other_ou["right_child_node_id"] == ou["plan_node_id"]:
                    ou[f"{prefix}_num_outer_loops"] = get_plan_rows_matching_plan_id(other_ou["left_child_node_id"])
                    ou[f"{prefix}_num_iterator_used"] = max(1.0, ou[f"{prefix}_num_iterator_used"] / ou[f"{prefix}_num_outer_loops"])
                    break
        elif ou_type == OperatingUnit.NestLoop:
            # Number of outer rows is the rows output by the left plan child.
            ou["NestLoop_num_outer_rows"] = get_plan_rows_matching_plan_id(ou["left_child_node_id"])
            ou["NestLoop_num_inner_rows_cumulative"] = get_plan_rows_matching_plan_id(ou["right_child_node_id"]) * ou["NestLoop_num_outer_rows"]

    new_ous.extend(ous)
    return new_ous


def generate_query_ous(conn, scratch, window_stats, idxoid_table_map, tableoid_map, metadata):
    tmp_data_dir = scratch / "ous"
    Path(tmp_data_dir).mkdir(parents=True, exist_ok=True)
    if Path(tmp_data_dir / "done").exists():
        return

    # Accumulator for all the OU features.
    ou_features = {ou.name: [] for ou in OperatingUnit}
    output_counter = 0
    num_added = 0

    def accumulate_ou(ou):
        nonlocal num_added
        nonlocal output_counter
        nonlocal ou_features
        ou_features[ou["node_type"]].append(ou)
        num_added += 1

        if num_added >= 1024 * 1024:
            for key in ou_features:
                if len(ou_features[key]) > 0:
                    filename = tmp_data_dir / f"{key}.feather.{output_counter}"
                    pd.DataFrame(ou_features[key]).to_feather(filename)
            ou_features = {ou.name: [] for ou in OperatingUnit}
            output_counter += 1
            num_added = 0

    key_fn = lambda x: int(x.split(".feather")[0].split("_")[-1])
    chunks = scratch / "chunks"
    chunk_files = sorted(glob.glob(f"{chunks}/chunk_*"), key=key_fn)
    ou_files = scratch / "ous"
    Path(ou_files).mkdir(parents=True, exist_ok=True)
    for chunk_file in chunk_files:
        logger.info("Processing query operating units for %s", chunk_file)
        chunk_stats = window_stats[key_fn(chunk_file)]
        chunk = pd.read_feather(chunk_file)

        pbar = tqdm(total=chunk.shape[0])
        for cgrp in chunk.groupby(by=["query_text"]):
            for i, tup in enumerate(cgrp[1].itertuples()):
                ous = evaluate_query(conn, cgrp[0])
                ous = augment_single_ou(conn, ous, tup, idxoid_table_map, tableoid_map, metadata, chunk_stats) 
                for ou in ous:
                    ou["query_id"] = tup.query_id
                    ou["query_order"] = tup.query_order
                    # FIXME(INTERVAL): what interval is this data executing in??
                    ou["unix_timestamp"] = float(0)
                    ou["PLOT_TS"] = tup.statement_timestamp
                    accumulate_ou(ou)
                pbar.update(1)
        pbar.close()

    for key in ou_features:
        if len(ou_features[key]) > 0:
            filename = tmp_data_dir / f"{key}.feather.{output_counter}"
            pd.DataFrame(ou_features[key]).to_feather(filename)

    open(tmp_data_dir / "done", "w").close()


def slice_windows(conn, input_dir, slice_window, TABLES, ff_tbl_change_map, pg_class, table_attr_map, table_keyspace_map, scratch, workload_model):
    # Load the initial data map.  data_map = {}
    data_map = {}
    for tbl in TABLES:
        if len(table_attr_map[tbl]) == 0:
            continue

        pks_sel = ",".join(table_attr_map[tbl])
        result = conn.execute(f"SELECT {pks_sel} FROM {tbl}", prepare=False)
        frame = pd.DataFrame(result, columns=table_attr_map[tbl])
        data_map[tbl] = frame

    # Get the initial state of the pgstattuple.
    table_state = {}
    for tbl in TABLES:
        result = [r for r in conn.execute(f"SELECT * FROM pgstattuple_approx('{tbl}')", prepare=False)][0]
        table_state[tbl] = {
                "table_len": result[0],
                "approx_free_percent": result[9],
                "dead_tuple_percent": result[7],
                "approx_tuple_count": result[2],
                "tuple_len_avg": result[3] / result[2],
                "ff": pg_class[tbl]["fillfactor"] * 100.0,
            }

    # Mutate the table state based on the queries.
    def mutate_table_state(tbl, table_state, queries, extend_percent, hot_percent):
        if queries.shape[0] == 0:
            return table_state

        num_insert = queries[queries.is_insert].num_modify.sum()
        num_delete = queries[queries.is_delete].num_modify.sum()
        num_update = queries[queries.OP == "UPDATE"].num_modify.sum()
        new_tuple_count = max(0, table_state["approx_tuple_count"] + num_insert - num_delete)

        num_hot = min(num_update, int(num_update * hot_percent))
        num_extend = extend_percent * (num_insert + num_update - num_hot)
        # Each update tuple incurs a somewhat "dead tuple".
        # But we have no idea about defrags...
        new_dead_tuples = (table_state["dead_tuple_percent"] * table_state["table_len"] / table_state["tuple_len_avg"]) + (num_delete + num_update)
        new_table_len = table_state["table_len"] + num_extend * 8192

        table_state["table_len"] = new_table_len
        table_state["approx_tuple_count"] = new_tuple_count
        # Dead tuple percent is relative to the total table len.
        table_state["dead_tuple_percent"] = new_dead_tuples / new_table_len
        # Approximate free is take the number of "available" and divide it by "# alive and # dead".
        table_state["approx_free_percent"] = max(0.0, 1 - (new_table_len / table_state["tuple_len_avg"]) / (new_tuple_count + new_dead_tuples))
        # FIXME(TUPLEN): Should we update the tuple length average? Assume it doesn't change, I guess. OR we'd have to estimate it. uh-oh.

        if tbl in ff_tbl_change_map:
            # Mutate the fill factor based on when we know the ALTER TABLE was executed.
            slots = ff_tbl_change_map[tbl]
            for (ts, ff) in slots:
                if np.sum(queries.statement_timestamp >= ts) > 0:
                    table_state["ff"] = ff
                    break
        return table_state

    chunks = scratch / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    if not Path(scratch / "chunk_stats").exists():
        loader = Loader(input_dir, slice_window, TABLES)
        chunk, augment_chunk = loader.get_next_slice()

        qnum = 0
        chunk_num = 0
        input_features = []
        window_stats = []
        while chunk is not None:
            logger.info("Processing chunk: %s with window size %s", chunk_num, slice_window)
            chunk["query_order"] = chunk.index + qnum
            chunk_window_stats = {}
            # Group by target from augment_chunk which is the keyspace defining chunk.
            for group in augment_chunk.groupby(by=["target"]):
                tbl = group[0]
                keyspace = []
                if tbl in table_keyspace_map and tbl in table_keyspace_map[tbl]:
                    keyspace = table_keyspace_map[tbl][tbl]

                data = data_map[group[0]].copy() if group[0] in data_map else None
                inputs = WorkloadModel.featurize(group[1], data, table_state[tbl], keyspace, train=False)
                inputs = workload_model.prepare_inputs(pd.DataFrame([inputs]), train=False)
                batch = next(iter(torch.utils.data.DataLoader(inputs, batch_size=1)))

                predictions = workload_model.predict(*batch)
                extend_percent, defrag_percent, hot_percent = predictions[0][0].item(), predictions[0][1].item(), predictions[0][2].item()
                chunk_window_stats[tbl] = {
                    "extend_percent": min(1.0, max(0.0, extend_percent)),
                    "defrag_percent": min(1.0, max(0.0, defrag_percent)),
                    "hot_percent": min(1.0, max(0.0, hot_percent)),
                }

                # Mutate the table state accordingly.
                table_state[tbl] = mutate_table_state(tbl, table_state[tbl], chunk[(chunk.OP != "SELECT") & (chunk.modify_target == tbl)], extend_percent, hot_percent)
            window_stats.append(chunk_window_stats)

            # Update the data map based on the modify target of the actual queries.
            for group in chunk.groupby(by=["modify_target"]):
                tbl = group[0]
                if tbl not in table_attr_map or tbl not in table_keyspace_map or tbl not in table_keyspace_map[tbl]:
                    # No valid keys that are worth looking at.
                    continue

                pk_keys = table_keyspace_map[tbl][tbl]
                all_keys = table_attr_map[tbl]
                data_map[tbl], _ = compute_frame(data_map[tbl], group[1], pk_keys, all_keys)

            # Write out the slice.
            chunk.to_feather(Path(chunks) / f"chunk_{chunk_num}.feather")
            augment_chunk.to_feather(Path(chunks) / f"augchunk_{chunk_num}.feather")
            qnum = qnum + chunk.shape[0]

            chunk_num = chunk_num + 1
            chunk, augment_chunk = loader.get_next_slice()

        with open((scratch / "chunk_stats"), "wb") as f:
            pickle.dump(window_stats, f)
    else:
        with open((scratch / "chunk_stats"), "rb") as f:
            window_stats = pickle.load(f)

    return window_stats


def main(benchmark, psycopg2_conn, session_sql, dir_models, dir_workload_model, dir_data, slice_window, dir_evals_output, dir_scratch, eval_batch_size):
    TABLES = BENCHDB_TO_TABLES[benchmark]

    # Load the models
    base_models = load_models(dir_models)
    workload_model = WorkloadModel()
    workload_model.load(dir_workload_model)

    # Scratch space is used to try and reduce debugging overhead.
    scratch = dir_scratch / "eval_query_workload_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    with open(f"{dir_data}/analysis/keyspaces.pickle", "rb") as f:
        table_attr_map = pickle.load(f)
        attr_table_map = pickle.load(f)
        table_keyspace_map = pickle.load(f)
        query_template_map = pickle.load(f)
        window_index_map = pickle.load(f)

    with psycopg.connect(psycopg2_conn, autocommit=True) as conn:
        conn.execute("SET qss_capture_enabled = OFF")
        if session_sql.exists():
            with open(session_sql, "r") as f:
                for line in f:
                    conn.execute(line)

        # Get metadata.
        metadata = prepare_pg_inference_state(conn)

        idxoid_table_map = {}
        result = conn.execute("""
            SELECT indexrelid,
                   t.relname
              FROM pg_index,
                   pg_class t
             WHERE pg_index.indrelid = t.oid
        """)
        for tup in result:
            idxoid_table_map[tup[0]] = tup[1]

        ddl = pd.read_csv(f"{dir_data}/pg_qss_ddl.csv")
        ddl = ddl[ddl.command == "AlterTableOptions"]
        ff_tbl_change_map = {t: [] for t in TABLES}
        for tbl in TABLES:
            query_str = ddl["query"].str
            slots = query_str.contains(tbl) & query_str.contains("fillfactor")
            tbl_changes = ddl[slots]
            for q in tbl_changes.itertuples():
                root = pglast.Node(pglast.parse_sql(q.query))
                for node in root.traverse():
                    if isinstance(node, pglast.node.Node):
                        if isinstance(node.ast_node, pglast.ast.DefElem):
                            if node.ast_node.defname == "fillfactor":
                                ff = node.ast_node.arg.val
                                ff_tbl_change_map[tbl].append((q.statement_timestamp, ff))

        window_stats = slice_windows(conn, dir_data, slice_window, TABLES, ff_tbl_change_map, metadata["pg_class"], table_attr_map, table_keyspace_map, scratch, workload_model)
        generate_query_ous(conn, scratch, window_stats, idxoid_table_map, metadata["pg_class_lookup"], metadata)
        augment_ous(scratch / "ous", [metadata], conn)

        key_fn = lambda x: int(x.split(".feather")[0].split("_")[-1])
        chunk_files = sorted(glob.glob(f"{scratch}/chunks/chunk_*"), key=key_fn)
        def read(f):
            df = pd.read_feather(f, columns=["query_id", "query_order", "query_text", "modify_target", "OP", "elapsed_us"])
            empty_target = df[df.modify_target.isnull()]
            for tbl in TABLES:
                mask = empty_target.query_text.str.contains(tbl)
                df.loc[empty_target[mask].index, "modify_target"] = tbl
            return df
        df = pd.concat(map(read, chunk_files), ignore_index=True, axis=0)

        eval_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_output = dir_evals_output / f"eval_{eval_timestamp}"
        base_output.mkdir(parents=True, exist_ok=True)
        evaluate_query_plans(eval_batch_size, base_models, scratch / "ous", df, base_output)
        with open(f"{base_output}/ddl_changes.pickle", "wb") as f:
            pickle.dump(ff_tbl_change_map, f)


class EvalQueryWorkloadCLI(cli.Application):
    benchmark = cli.SwitchAttr(
        "--benchmark",
        str,
        mandatory=True,
        help="Benchmark that is being evaluated.",
    )
    session_sql = cli.SwitchAttr(
        "--session-sql",
        Path,
        mandatory=False,
        help="Path to a list of SQL statements that should be executed in the session prior to EXPLAIN.",
    )
    dir_data = cli.SwitchAttr(
        "--dir-data",
        Path,
        mandatory=True,
        help="Folder containing raw evaluation CSVs.",
    )
    dir_evals_output = cli.SwitchAttr(
        "--dir-evals-output",
        Path,
        mandatory=True,
        help="Folder to output evaluations to.",
    )
    dir_scratch = cli.SwitchAttr(
        "--dir-scratch",
        Path,
        mandatory=False,
        help="Folder to use as scratch space.",
        default = Path("/tmp/"),
    )
    dir_models = cli.SwitchAttr(
        "--dir-base-models",
        Path,
        mandatory=True,
        help="Folder containing the base evaluation models.",
    )
    dir_workload_model = cli.SwitchAttr(
        "--dir-workload-model",
        Path,
        mandatory=True,
        help="Folder containing the workload model.",
    )
    psycopg2_conn = cli.SwitchAttr(
        "--psycopg2-conn",
        mandatory=True,
        help="Psycopg2 connection string for connecting to a valid database.",
    )
    slice_window = cli.SwitchAttr(
        "--slice-window",
        int,
        mandatory=True,
        help="Size of the window slice that should be used.",
    )
    eval_batch_size = cli.SwitchAttr(
        "--eval-batch-size",
        int,
        default=4,
        help="Number of OU files to evaluate in a batch.",
    )


    def main(self):
        main(self.benchmark,
             self.psycopg2_conn,
             self.session_sql,
             self.dir_models,
             self.dir_workload_model,
             self.dir_data,
             self.slice_window,
             self.dir_evals_output,
             self.dir_scratch,
             self.eval_batch_size)


if __name__ == "__main__":
    EvalQueryWorkloadCLI.run()
