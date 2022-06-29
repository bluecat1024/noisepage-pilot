import random
import gc
import re
import copy
import json
import math
import psycopg
from tqdm import tqdm
from psycopg.rows import dict_row
from distutils import util
from pathlib import Path
from enum import Enum, auto, unique
import pandas as pd
import numpy as np
from behavior import OperatingUnit
from behavior.plans import DIFF_SCHEMA_METADATA
from behavior.modeling import featurize
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


BLOCKSZ = 8192
QUERY_CACHE = {}


def _est_tuple_key_size(tuple_size):
    # 32 bytes of padding (23 byte header, possible null-bitmap, 4 byte TID..)
    return tuple_size + 32


def _est_index_key_size(key_size):
    # 16 bytes of padding (ItemId + IndexTuple header and various elements)
    return key_size + 16


def _est_available_page_size():
    # Take a fixed size header (https://www.postgresql.org/docs/current/storage-page-layout.html).
    # TODO(wz2): We only estimate the impact of header on the page size, not special data/extra fields.
    return BLOCKSZ - 24


def _implant_stats(conn, pg_class_stats):
    conn.execute("SELECT qss_clear_stats()")

    for obj in pg_class_stats.keys():
        stats = pg_class_stats[obj]
        relpages = stats['relpages']
        reltuples = stats['reltuples']
        height = 0

        if stats['relkind'] == 'i':
            # This an index so use BLOCKSZ / est_key_size
            fanout = _est_available_page_size() / (_est_index_key_size(stats['est_key_size']))
            fanout = 1 if fanout == 0 else fanout
            height = math.ceil(np.log(relpages) / np.log(fanout))
        query = f"SELECT qss_install_stats('{obj}', {relpages}, {reltuples}, {height})"
        conn.execute(query)


def _evaluate_query_for_plan(conn, query_str):
    global QUERY_CACHE
    if query_str in QUERY_CACHE and QUERY_CACHE[query_str] is not None:
        return copy.deepcopy(QUERY_CACHE[query_str])
    else:
        # Execute EXPLAIN (format tscout) to get all the features that we want.
        result = [r for r in conn.execute("EXPLAIN (format tscout) " + query_str, prepare=False)]
        result = result[0][0]

        # Extract all the OUs for a given query_text plan.
        features = json.loads(result)[0]
        template_ous = []

        def extract_ou(plan):
            ou = {}
            for key in plan:
                value = plan[key]
                if key == "Plans":
                    for p in plan[key]:
                        extract_ou(p)
                        continue
                if isinstance(value, list):
                    # TODO(wz2): For now, we simply featurize a list[] with a numeric length.
                    # This is likely insufficient if the content of the list matters significantly.
                    ou[key + "_len"] = len(value)
                ou[key] = value
            ou["query_text"] = query_str
            template_ous.append(ou)
        extract_ou(features)

        if query_str in QUERY_CACHE:
            # Only store queries that will hit again (i.e., repeat).
            QUERY_CACHE[query_str] = copy.deepcopy(template_ous)
        else:
            QUERY_CACHE[query_str] = None
        return template_ous


def _guess_single_split_index(idx_metadata):
    idx_key_size = _est_index_key_size(idx_metadata["est_key_size"])
    keys_per_page = (_est_available_page_size()) / idx_key_size
    split_threshold = keys_per_page / 2;

    # TODO(wz2): This is a rough approximation at when upper level tree splits happen. We also generally just assume
    # that whenever we insert half the page, we will trigger a page split.
    num_splits = 0
    while (idx_metadata['sim_inserts'] % split_threshold) == 0:
        num_splits += 1
        split_threshold *= split_threshold

    return num_splits


def _guess_extend_relation(tbl_metadata, total_inserts):
    fill_factor = tbl_metadata['fillfactor']
    num_tuples_per_page = math.floor((_est_available_page_size() * fill_factor) / _est_tuple_key_size(tbl_metadata['est_tuple_size']))
    new_free_slots = math.floor((_est_available_page_size() * (1 - fill_factor) / _est_tuple_key_size(tbl_metadata['est_tuple_size'])))

    old_new_pages = math.ceil(tbl_metadata["sim_inserts"] / num_tuples_per_page) + 1
    new_pages = math.ceil((tbl_metadata["sim_inserts"] + total_inserts) / num_tuples_per_page) + 1

    # TODO(wz2): We return new_free_slots and once again make the assumption that they are distributed
    # uniformally (when we know they aren't bonk).
    return (new_pages - old_new_pages), (new_pages - old_new_pages) * new_free_slots


def _guess_inserts_till_extend_relation(tbl_metadata):
    fill_factor = tbl_metadata['fillfactor']
    num_tuples_per_page = math.floor((_est_available_page_size() * fill_factor) / _est_tuple_key_size(tbl_metadata['est_tuple_size']))
    assert num_tuples_per_page > 0
    return num_tuples_per_page - (tbl_metadata["sim_inserts"] % num_tuples_per_page)


def _guess_hot_update_slot(tbl_metadata, num_updates):
    # TODO(wz2): This fundamentally banks on uniformity (live tuples spread, free slot spread, access spread).
    # As a rough model, until we hit # free slots / # pages <= 1, we just assume that it will HOT. Under uniformity,
    # given 2 free slots / page and [# page] updates, we should expect all updates to be HOT.

    num_hot = 0
    if tbl_metadata["relpages"] == 1:
        # Special case where all the free slots are on the same page.
        return min(tbl_metadata["sim_free_slots_update"], num_updates)
    elif (tbl_metadata["sim_free_slots_update"] - num_updates) > tbl_metadata["relpages"]:
        return num_updates
    elif tbl_metadata["sim_free_slots_update"] > tbl_metadata["relpages"]:
        # These are the ones that can be HOT-able.
        num_hot = tbl_metadata["sim_free_slots_update"] - tbl_metadata["relpages"]
        num_updates -= num_hot

    update_slots = tbl_metadata["sim_free_slots_update"]
    for _ in range(int(num_updates)):
        coin_flip_threshold = update_slots / float(tbl_metadata["relpages"])
        hot = int(random.uniform(0, 1) < coin_flip_threshold)

        num_hot += hot
        update_slots -= hot

    return num_hot


def _guess_relation_mutation(tbl_metadata, num_inserts, num_updates):
    # We can only have X number of inserts or updates, but not both at a time for a mutation.
    # This is because a new page extension can effect the outcome of future inserts/updates.
    assert (num_inserts == 0 and num_updates > 0) or (num_inserts > 0 and num_updates == 0)

    # Compute the number of HOT updates that will ensue.
    num_hot = 0 if num_updates == 0 else _guess_hot_update_slot(tbl_metadata, num_updates)
    num_slots_used = 0
    num_new_update_slots = 0
    num_new_pages = 0
    tbl_metadata["sim_free_slots_update"] -= num_hot

    # TODO(wz2): First try to address the inserts, and then use those new update slots to address
    # updates. This is not fully correct but is a better approximation.

    # These are how many slots that go to extend the relation.
    remaining = num_inserts + (num_updates - num_hot)
    if remaining <= tbl_metadata["sim_free_slots"]:
        # This case is where all remaining actions can be addressed by # of free slots.
        # This case triggers no inserts.
        num_slots_used = remaining
        tbl_metadata["sim_free_slots"] -= num_slots_used
    else:
        # We've used the remaining allocation of free slots.
        num_slots_used = tbl_metadata["sim_free_slots"]
        tbl_metadata["sim_free_slots"] = 0
        # These are how many more slots we need.
        num_inserts = remaining - num_slots_used
        inserts_till_extend = _guess_inserts_till_extend_relation(tbl_metadata)

        if num_inserts > inserts_till_extend and num_updates > 0:
            # In the update case where we need to account for new slot production.
            # First insert up until we produce a new page.
            num_new_pages, num_new_update_slots = _guess_extend_relation(tbl_metadata, inserts_till_extend)
            assert num_new_pages > 0

            # Update the sim_free_slots_update counter to reflect this.
            tbl_metadata["sim_free_slots_update"] += num_new_update_slots
            # Mark that we've inserted inserts_till_extend.
            tbl_metadata["sim_inserts"] += inserts_till_extend

            # Recursively evaluate the remaining portion and accumulate final stats.
            nnp, nh, nnus, ni, nsu = _guess_relation_mutation(tbl_metadata, 0, num_inserts - inserts_till_extend)
            num_new_pages += nnp
            num_hot += nh
            num_new_update_slots += nnus
            num_inserts += ni
            num_slots_used += nsu
        else:
            # This is the case where for INSERT: we just produce new pages.
            # For UPDATE, either no UPDATE produces a new page or the last tuple produces one.
            num_new_pages, num_new_update_slots = _guess_extend_relation(tbl_metadata, num_inserts)
            tbl_metadata["sim_free_slots_update"] += num_new_update_slots
            tbl_metadata["sim_inserts"] += num_inserts

    return num_new_pages, num_hot, num_new_update_slots, num_inserts, num_slots_used


def _model_vacuum_runtime(prev_table_stats, pg_settings):
    # TODO(wz2): Maybe this should not use a formulaic model -> but instead should be a true model?
    autovac_delay = pg_settings["autovacuum_vacuum_cost_delay"] * 1000.0
    hit_cost = pg_settings["vacuum_cost_page_hit"]
    miss_cost = pg_settings["vacuum_cost_page_miss"]
    dirty_cost = pg_settings["vacuum_cost_page_dirty"]
    cost_limit = pg_settings["vacuum_cost_limit"] if pg_settings["autovacuum_vacuum_cost_limit"] == -1 else pg_settings["autovacuum_vacuum_cost_limit"]

    # Postgres has a buffer pool of 256 KB for VACUUM. We actually have no idea where the UPDATE/DELETE
    # tuple is going to be routed and so all we can do is make an assumption that n_dead_tup are dirty.
    # From that reasoning, we also assume that the first buffer pool "cycle" will all hit and rest will miss.

    # TODO(wz2): Assume that dirty-ing tuples is uniform. This is consistent with the assumption that holes are
    # produced uniformally across the blocks. How UPDATE and VACUUM models work need to be consistent.
    dirty = min(prev_table_stats["n_dead_tup"], prev_table_stats["relpages"])
    hit = (prev_table_stats["relpages"]) if prev_table_stats["relpages"] < 32 else 32
    miss = (prev_table_stats["relpages"] - 32) if (prev_table_stats["relpages"] - 32) > 0 else 0
    total_cost = (dirty * dirty_cost) + (hit * hit_cost) + (miss * miss_cost)

    # TODO(wz2): Probably need some form of benchmark or microbenchmark to get an accurate estimate.
    DIRTY_FSYNC_TIME_US = 0.0
    READ_TIME_US = 0.0
    autovac_delay_time = math.floor(total_cost / cost_limit) * autovac_delay
    dirty_fsync_time = dirty * DIRTY_FSYNC_TIME_US
    miss_time = miss * READ_TIME_US
    return autovac_delay_time + dirty_fsync_time + miss_time


def _compute_next_window_statistics(query_stream_slice, previous_stats, tables_process):
    new_stats = copy.deepcopy(previous_stats)
    last_query = query_stream_slice.iloc[-1]

    autovac_runtime = 0
    for tbl in tables_process:
        delta_inserts = last_query[f"{tbl}_cum_inserts"] - previous_stats["pg_class"][tbl]["prev_cum_inserts"]
        delta_updates = last_query[f"{tbl}_cum_updates"] - previous_stats["pg_class"][tbl]["prev_cum_updates"]
        delta_deletes = last_query[f"{tbl}_cum_deletes"] - previous_stats["pg_class"][tbl]["prev_cum_deletes"]
        delta_updels = delta_updates + delta_deletes

        # We do this so the relation mutation is reasonably consistent with how we determine which OU
        # gets hit with the relation extension later on in the pipeline. Some of this code is sad.
        # This also conveniently puts all the sim_* counters into the right place.
        num_new_pages = 0
        targets = query_stream_slice.modify_target == tbl
        targets &= ((query_stream_slice["is_insert"] == 1) | (query_stream_slice["is_update"] == 1))
        query_slice = query_stream_slice[targets]
        for query in query_slice.itertuples():
            ninsert = query.num_modify if query.is_insert else 0
            nupdates = query.num_modify if ninsert == 0 else 0
            npage, _, _, _, _ = _guess_relation_mutation(new_stats["pg_class"][tbl], ninsert, nupdates)
            num_new_pages += npage

        # This is about logical tuples to the relation.
        new_stats["pg_class"][tbl]["reltuples"] = previous_stats["pg_class"][tbl]["reltuples"] + delta_inserts - delta_deletes
        # Update the estimate on how many new pages there will be after executing the segment.
        new_stats["pg_class"][tbl]["relpages"] += num_new_pages

        # TODO(wz2): Updating the cardinality/selectivity of ANALYZE is a whole other beast.
        # We don't address that. We don't particularly foresee those having a huge impact either but may be wrong~

        # Model and attempt to estimate the runtime of VACUUM.
        autovac_runtime += _model_vacuum_runtime(previous_stats["pg_class"][tbl], previous_stats["pg_settings"])

        # The number of dead tuples in the previous round get cleaned up and opened up.
        new_stats["pg_class"][tbl]["sim_free_slots"] += previous_stats["pg_class"][tbl]["n_dead_tup"]

        # Updates contribute to dead tuples because the old version is no longer visible.
        new_stats["pg_class"][tbl]["n_dead_tup"] = delta_updels

        # TODO(wz2): What about indexes?
        for index in new_stats["pg_class"][tbl]["indexes"]:
            index_stats = new_stats["pg_class"][index]
            # Always install the new tuples and dead tuple counts as the new tuples. Indexes still have pointers.
            index_stats["reltuples"] = new_stats["pg_class"][tbl]["reltuples"] + new_stats["pg_class"][tbl]["n_dead_tup"]

            for _ in range(delta_inserts + delta_updates): 
                index_stats["sim_inserts"] += 1
                index_stats["relpages"] += _guess_single_split_index(index_stats)

        # Update the thresholds.
        new_stats["pg_class"][tbl]["prev_cum_inserts"] = last_query[f"{tbl}_cum_inserts"]
        new_stats["pg_class"][tbl]["prev_cum_updates"] = last_query[f"{tbl}_cum_updates"]
        new_stats["pg_class"][tbl]["prev_cum_deletes"] = last_query[f"{tbl}_cum_deletes"]
        new_stats["pg_class"][tbl]["prev_cum_updels"] = last_query[f"{tbl}_cum_updates"] + last_query[f"{tbl}_cum_deletes"]

    return new_stats, autovac_runtime


def _simulate_modify_table(conn, ou, template_ous, metadata):
    add_ous = []
    ou_type = OperatingUnit[ou["node_type"]]
    relname = metadata["pg_class_lookup"][ou["ModifyTable_target_oid"]]
    tbl_metadata = metadata["pg_class"][relname]
    if ou_type == OperatingUnit.ModifyTableInsert or ou_type == OperatingUnit.ModifyTableUpdate:
        num_insert = 1 if ou_type == OperatingUnit.ModifyTableInsert else 0
        num_update = ou["ModifyTableUpdate_num_updates"] if ou_type == OperatingUnit.ModifyTableUpdate else 0
        num_new_pages, num_hot, _, _, _ = _guess_relation_mutation(tbl_metadata, num_insert, num_update)
        assert num_hot <= num_update
        ou[f"{ou_type.name}_num_extends"] = num_new_pages

        for _ in range(int((num_insert + num_update) - num_hot)):
            for idxoid in ou["ModifyTable_indexupdates_oids"]:
                idx_name = metadata['pg_class_lookup'][idxoid]
                idx_metadata = metadata['pg_class'][idx_name]
                idx_metadata['sim_inserts'] += 1

                num_splits = _guess_single_split_index(idx_metadata)
                add_ous.append({
                    'node_type': OperatingUnit.ModifyTableIndexInsert.name,
                    'ModifyTableIndexInsert_indexid': idxoid,
                    'plan_node_id': -1,
                    # TODO(wz2): We assume that a split will always cause an extension (in the general case, if a page
                    # is completely free, we will try and re-use that page first).
                    'ModifyTableIndexInsert_num_extends': num_splits,
                    'ModifyTableIndexInsert_num_splits': num_splits,
                })

    for tgoid in ou["ModifyTable_ar_triggers"]:
        trigger_info = metadata['pg_trigger'][tgoid]
        if trigger_info['contype'] != 'f':
            # UNIQUE constraints should be handled by indexes.
            continue

        if ou_type == OperatingUnit.ModifyTableInsert:
            # 1644 is the hardcoded code for RI_FKey_check_ins.
            assert trigger_info["tgfoid"] == 1644
            frelname = metadata["pg_class_lookup"][trigger_info["confrelid"]]
            query = f"SELECT 1 FROM {frelname} WHERE "
            for i, eqop in enumerate(trigger_info["conpfeqop"]):
                assert eqop in (15, 91, 92, 93, 94, 96, 98, 254, 260, 352, 353, 5068, 385, 387, 410, 416,
                                503, 532, 533, 607, 649, 620, 670, 792, 900, 974, 1054, 1070, 1093, 1108,
                                1550, 1120, 1130, 1320, 1330, 1500, 1535, 1616, 1220, 3362, 1201, 1752,
                                1784, 1804, 1862, 1868)
                
            for i, attnum in enumerate(trigger_info["confkey"]):
                search = (trigger_info["confrelid"], attnum)
                att_info = metadata["pg_attribute"][search]
                if i != 0:
                    query = query + " AND "
                # TODO(wz2): We currently use a placeholder value. Implication is that for certain cardinalities, this is bust.
                query = query + f"{att_info['attname']} = '0'";

            # Get the OUs for the trigger query plan and compute the derived OUs.
            ous = _evaluate_query_for_plan(conn, query)
            for ou in ous:
                derived_ous = _compute_derived_ous(conn, ou, ous, metadata)
                # Wipe out the plan node ID.
                for ou in derived_ous:
                    ou["plan_node_id"] = -1
                    ou["query_text"] = query
                add_ous.extend(derived_ous)
        elif ou_type == OperatingUnit.ModifyTableUpdate:
            # Assert that the UPDATE/DELETE is basically a no-op
            assert trigger_info['confupdtype'] == 'a'
        else:
            assert ou_type == OperatingUnit.ModifyTableDelete
            assert trigger_info['confdeltype'] == 'a'

    add_ous.append(ou)
    return add_ous


def _compute_derived_ous(conn, ou, template_ous, metadata):
    # TODO(wz2): Here we compute derived OU features. In reality, we should evaluate whether
    # we need to actually compute these derived OU features OR whether we could just replace
    # actual plan feature estimates with these features during training.

    def get_key(key, other):
        for subst_key in other:
            if key in subst_key:
                return other[subst_key]
        assert False, f"Could not find {key} in {other}"

    def get_plan_rows_matching_plan_id(plan_id):
        for target_ou in template_ous:
            if target_ou["plan_node_id"] == plan_id:
                return get_key("plan_plan_rows", target_ou)
        assert False, f"Could not find plan node with {plan_id}"

    add_ous = []
    ou_type = OperatingUnit[ou["node_type"]]
    if ou_type == OperatingUnit.IndexOnlyScan or ou_type == OperatingUnit.IndexScan:
        prefix = "IndexOnlyScan" if ou_type == OperatingUnit.IndexOnlyScan else "IndexScan"

        # IndexOnlyScan/IndexScan need both num_iterator_used and num_outer_loops (for NestLoop).
        ou[f"{prefix}_num_iterator_used"] = get_key("plan_rows", ou)
        for other_ou in template_ous:
            if other_ou["node_type"] == OperatingUnit.NestLoop.name and other_ou["right_child_node_id"] == ou["plan_node_id"]:
                ou[f"{prefix}_num_outer_loops"] = get_plan_rows_matching_plan_id(other_ou["left_child_node_id"])
                break

    elif ou_type == OperatingUnit.ModifyTableInsert or ou_type == OperatingUnit.ModifyTableUpdate or ou_type == OperatingUnit.ModifyTableDelete:
        if ou_type == OperatingUnit.ModifyTableUpdate:
            ou["ModifyTableUpdate_num_updates"] = get_key("ModifyTable_input_plan_rows", ou)
        add_ous.extend(_simulate_modify_table(conn, ou, template_ous, metadata))

    elif ou_type == OperatingUnit.Agg:
        # The number of input rows is the number of rows output by the child.
        ou["Agg_num_input_rows"] = get_plan_rows_matching_plan_id(ou["left_child_node_id"])

    elif ou_type == OperatingUnit.NestLoop:
        # Number of outer rows is the rows output by the left plan child.
        ou["NestLoop_num_outer_rows"] = get_plan_rows_matching_plan_id(ou["left_child_node_id"])
        # TODO(wz2): Use the cardinality estimator on the right child -> NestLoop output is controlled by join condition.
        # This does assume that inner probe returns same # tuples given an outer probe.
        ou["NestLoop_num_inner_rows_cumulative"] = get_plan_rows_matching_plan_id(ou["right_child_node_id"]) * ou["NestLoop_num_outer_rows"]

    add_ous.append(ou)
    return add_ous


def generate_query_ous(window_queries, window_metadata, conn, tmp_data_dir):
    global QUERY_CACHE

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

    for i, (queries, metadata) in enumerate(zip(window_queries, window_metadata)):
        QUERY_CACHE = {}
        gc.collect()

        # Sort by order so we get an itertuples() in execution order.
        queries.set_index(keys=["order"], drop=True, append=False, inplace=True)
        queries.sort_index()

        # Do a JEDI mind-trick on the postgres database for pg_class statistics.
        _implant_stats(conn, metadata['pg_class'])

        for query in tqdm(queries.itertuples(), total=queries.shape[0]):
            # Yoink all the OUs for a given query plan.
            plan_ous = []
            ous = _evaluate_query_for_plan(conn, query.query_text)
            for ou in ous:
                plan_ous.extend(_compute_derived_ous(conn, ou, ous, metadata))

            # Fill in relevant OU metadata for evaluation/combining later.
            for ou in plan_ous:
                ou["query_id"] = query.query_id

                # This should be query.order
                ou["order"] = query.Index

                # This is used for augmentation purposes.
                ou["unix_timestamp"] = float(i)
                assert "plan_node_id" in ou
                accumulate_ou(ou)

                oukind = OperatingUnit[ou["node_type"]]
                if oukind == OperatingUnit.ModifyTableInsert or oukind == OperatingUnit.ModifyTableUpdate or oukind == OperatingUnit.ModifyTableDelete:
                    # Use the estimate for the number of input tuples as the number we will modify.
                    queries.at[query.Index, "num_modify"] = ou["ModifyTable_input_plan_rows"]

        # Reset the order index out.
        queries.reset_index(drop=False, inplace=True)

    QUERY_CACHE = {}
    gc.collect()

    for key in ou_features:
        if len(ou_features[key]) > 0:
            filename = tmp_data_dir / f"{key}.feather.{output_counter}"
            pd.DataFrame(ou_features[key]).to_feather(filename)

    # Create the output marker file.
    open(tmp_data_dir / "generate_query_ous", "w").close()


def generate_vacuum_partition(query_stream, metadata):
    # Sort the query_stream dataframe in order of execution order.
    query_stream.set_index(keys=["order"], drop=True, append=False, inplace=True)
    query_stream.sort_index(axis=0, inplace=True)
    query_stream.reset_index(drop=False, inplace=True)

    # These are the slices that should be returned.
    query_slices = []
    metadata_slices = [metadata]
    pg_settings = metadata["pg_settings"]

    for relname in metadata["pg_class"].keys():
        if np.sum(query_stream.modify_target == relname) == 0:
            # No one uses this table.
            continue

        query_stream[f"{relname}_cum_inserts"] = ((query_stream.modify_target == relname) * query_stream.is_insert * query_stream.num_modify).cumsum()
        query_stream[f"{relname}_cum_updates"] = ((query_stream.modify_target == relname) * query_stream.is_update * query_stream.num_modify).cumsum()
        query_stream[f"{relname}_cum_deletes"] = ((query_stream.modify_target == relname) * query_stream.is_delete * query_stream.num_modify).cumsum()

    # TODO(wz2): Assume queries are executed one after the other with no think time.
    # It perhaps might be worth modeling a <think time> in certain cases.
    # Might just be some P(time to next query | last query or current txn step)
    query_stream["cum_start_time_us"] = query_stream.pred_elapsed_us.cumsum()

    # TODO(wz2): Assume that auto VACUUM wakes up when the workload starts running. This is generally
    # not the case but since we don't have true wall-clock time, we have to do this.
    current_vacuum_time = 0
    while query_stream.shape[0] > 0 and current_vacuum_time < query_stream.iloc[-1]["cum_start_time_us"]:
        # Find when the next VACUUM will run that will actually trigger a change.
        tables_process = []

        # Compute the earliest next time that autovacuum can execute.
        target_time = current_vacuum_time + pg_settings["autovacuum_naptime"] * 1000

        # Get the current page statistics that we should use in determining a target.
        prev_metadata = copy.deepcopy(metadata_slices[-1])

        for relname in prev_metadata["pg_class"].keys():
            if prev_metadata["pg_class"][relname]["relkind"] != 'r':
                # Ignore indexes since autovacuum fires on tables and then affects indexes.
                continue

            ins_target = prev_metadata["pg_class"][relname]["reltuples"] * pg_settings["autovacuum_vacuum_insert_scale_factor"] + pg_settings["autovacuum_vacuum_insert_threshold"]
            updel_target = prev_metadata["pg_class"][relname]["reltuples"] * pg_settings["autovacuum_vacuum_scale_factor"] + pg_settings["autovacuum_vacuum_threshold"]

            # We want to pad the targets with the targets from the last auto VACUUM. Why?
            # This is because the inserts/updates/deletes are cumulative across the entire workload.
            ins_target = ins_target + prev_metadata["pg_class"][relname]["prev_cum_inserts"]
            updel_target = updel_target + prev_metadata["pg_class"][relname]["prev_cum_updels"]

            # Slots where the INSERT or UPDATE/DELETE target have been met.
            slice_stream = query_stream[query_stream.cum_start_time_us <= target_time]
            insert_counts = slice_stream[f"{relname}_cum_inserts"] >= ins_target
            updels_counts = (slice_stream[f"{relname}_cum_updates"] + slice_stream[f"{relname}_cum_deletes"]) >= updel_target

            # If there is a valid slot where we are within the timebounds and have met the targets,
            # then we will vacuum this table.
            if len(slice_stream[(insert_counts | updels_counts)]) > 0:
                tables_process.append(relname)

        if len(tables_process) > 0:
            # Compute the next VACUUM window's statistics over the course of this "time-slice".
            new_stats, runtime = _compute_next_window_statistics(query_stream[query_stream.cum_start_time_us <= target_time], prev_metadata, tables_process)

            # TODO(wz2): Here we generally assume that AUTOVACUUM starts at the tail end of some query.
            # For queries that execute while AUTOVACUUM is running, we assume that they don't get charged,
            # and will execute under the previous snapshot window.

            # As such, we append all queries that fall under [AUTOVACUUM_START, AUTOVACUUM_END] to the "current" vacuum window.
            query_slices.append(query_stream[query_stream.cum_start_time_us <= target_time + runtime].copy(deep=True))
            query_stream = query_stream[query_stream.cum_start_time_us > target_time + runtime]
            metadata_slices.append(new_stats)
            target_time += runtime

        # Advance the vacuum time.
        current_vacuum_time = target_time

    if len(query_stream) > 0:
        # Append the last chunk of data. We already computed the stats previously.
        query_slices.append(query_stream)
    else:
        # Eliminate the last query_stats in the list (since there's no query to use that window).
        metadata_slices = metadata_slices[:len(metadata_slices)-1]

    for query_slice in query_slices:
        drop_labels = [col for col in query_slice.columns if any(x in col for x in ["cum_inserts", "cum_updates", "cum_deletes"])]
        query_slice.drop(labels=drop_labels, axis=1, inplace=True)
        query_slice.reset_index(drop=True, inplace=True)

    assert len(query_slices) == len(metadata_slices)
    return query_slices, metadata_slices


def estimate_query_modifications(raw_query_stream, initial_metadata, skip_query=False):
    if not skip_query:
        query_text = raw_query_stream.query_text.str.lower()
        insert_vector = query_text.str.startswith('insert')
        update_vector = query_text.str.startswith('update')
        delete_vector = query_text.str.startswith('delete')
        raw_query_stream["pred_elapsed_us"] = 0.0
        raw_query_stream["cum_start_time_us"] = 0.0
        raw_query_stream["modify_target"] = ""
        # Initially seed the number of modify entries as 1.
        raw_query_stream["num_modify"] = (insert_vector | update_vector | delete_vector).astype(int)
        raw_query_stream["is_insert"] = (insert_vector).astype(int)
        raw_query_stream["is_update"] = (update_vector).astype(int)
        raw_query_stream["is_delete"] = (delete_vector).astype(int)

        for key in initial_metadata["pg_class"].keys():
            key_contains = query_text.str.contains(key)
            raw_query_stream.loc[key_contains, "modify_target"] = key

    for key in initial_metadata["pg_class"].keys():
        tbl_metadata = initial_metadata["pg_class"][key]
        if tbl_metadata["relkind"] == "i":
            # Below computation only relevant for tables for now.
            tbl_metadata["sim_inserts"] = 0
            continue

        tbl_metadata["prev_cum_inserts"] = 0
        tbl_metadata["prev_cum_updates"] = 0
        tbl_metadata["prev_cum_deletes"] = 0
        tbl_metadata["prev_cum_updels"] = 0

        tbl_metadata["sim_inserts"] = 0
        if "fillfactor" not in tbl_metadata:
            tbl_metadata["fillfactor"] = 1.0
        fillfactor = tbl_metadata["fillfactor"]

        tsize = _est_tuple_key_size(tbl_metadata["est_tuple_size"])
        # This computes the total number of slots that can be allocated to the relation.
        all_slots = (_est_available_page_size() * tbl_metadata["relpages"] / tsize)
        # TODO(wz2): Once again, we assume here that all pages have roughly uniform free page slots
        # and slots that are in-use by live or still-yet-to-be-cleaned tuples.

        # This computes the number of slots that can be allocated to new appends.
        usable_insert_slots = all_slots * fillfactor
        usable_update_slots = all_slots * (1 - fillfactor)
        sim_free_slots = 0
        sim_free_slots_update = 0
        if usable_insert_slots > tbl_metadata["est_valid_slots"]:
            # This computes the number of free slots that can be used by new appends. We can only do additional
            # heap inserts if there are more usable insert slots than valid insert slots.
            sim_free_slots = usable_insert_slots - tbl_metadata["est_valid_slots"]
            sim_free_slots_update = usable_update_slots
        else:
            # If we have more valid tuples than usable insert slots, that needs to be accounted for in HOT update
            # budget so we subtract that count of tuples.
            usable_update_slots -= (tbl_metadata["est_valid_slots"] - usable_insert_slots)
            sim_free_slots_update = usable_update_slots if usable_update_slots > 0 else 0
        tbl_metadata["sim_free_slots"] = int(sim_free_slots)
        tbl_metadata["sim_free_slots_update"] = int(sim_free_slots_update)

        tbl_metadata.pop("est_valid_slots", None)
        tbl_metadata.pop("est_relation_size", None)

    return raw_query_stream, initial_metadata
