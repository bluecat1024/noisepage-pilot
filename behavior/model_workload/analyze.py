import glob
import pickle
import pglast
import psycopg
from tqdm import tqdm
import pandas as pd
from plumbum import cli
from pathlib import Path
import shutil
import numpy as np
import json
import logging

from behavior import BENCHDB_TO_TABLES

# Queries to block.
QUERY_CONTENT_BLOCK = [
    "pg_",
    "version()",
    "current_schema()",
    "pgstattuple_approx",
]

PER_QUERY_INDEX = ["query_id", "db_id", "pid", "statement_timestamp"]
AUX_COLUMNS = ["plan_node_id", "elapsed_us", "payload", "comment", "txn", "target", "num_rel_refs", "query_text", "generation", "id"]
TABLES = []

QUERY_TABLE_ATTRIBUTE_COLUMNS = """
select
    t.relname as table_name,
    a.attname as attname
from
    pg_class t,
    pg_attribute a
where
    a.attrelid = t.oid
    and a.attnum > 0
    and t.relkind = 'r'
"""

QUERY_TABLE_INDEX_COLUMNS = """
select
    t.relname as table_name,
    i.relname as index_name,
    ix.indexrelid,
    ix.indisprimary,
    array_to_string(array_agg(a.attname), ',') as column_names
from
    pg_class t,
    pg_class i,
    pg_index ix,
    pg_attribute a
where
    t.oid = ix.indrelid
    and i.oid = ix.indexrelid
    and a.attrelid = t.oid
    and a.attnum = ANY(ix.indkey)
    and t.relkind = 'r'
group by
    t.relname,
    i.relname,
    ix.indexrelid,
    ix.indisprimary
order by
    t.relname,
    i.relname;
"""

logger = logging.getLogger("workload_analyze")


def handle_inserts(process, roots):
    # Handle the case of ModifyTableInsert.
    inserts = process[process.comment == 'ModifyTableInsert']
    inserts = inserts.reset_index(drop=True)
    inserts.set_index(PER_QUERY_INDEX, inplace=True)

    # FIXME(INSERT): Assume that we are only inserting 1 tuple at a time.
    # counter4 is ModifyTableInsert_num_extends.
    inserts["num_modify"] = 1
    inserts["num_extend"] = inserts.counter4
    inserts["target_insert"] = inserts.target
    inserts.drop(columns=[f"counter{i}" for i in range(0, 10)] + AUX_COLUMNS, inplace=True)
    roots = roots.join(inserts, how='left')
    roots.fillna(value={"num_modify": 0, "num_extend": 0}, inplace=True)

    useful_vec = roots.index.isin(inserts.index)
    roots.loc[useful_vec, "target"] = roots.loc[useful_vec, "target_insert"]
    roots.drop(columns=["target_insert"], inplace=True)
    del inserts
    return roots


def handle_idxscans(process, roots, indexoid_table_map):
    # Handle the case of IndexScan to get defrags.
    # FIXME(wz2): Assume that only IndexScans can trigger a prune/defrag operation.
    idx_scans = process[(process.comment == "IndexScan") | (process.comment == "IndexOnlyScan")]
    idx_scans = idx_scans.reset_index(drop=True)
    idx_scans["target_scan"] = idx_scans.payload.apply(lambda x: indexoid_table_map[x] if x in indexoid_table_map else None)
    # counter3 is Index[Only]Scan_num_defrag.
    idx_scans["num_defrag"] = idx_scans.counter3
    idx_scans["num_modify_idx"] = idx_scans.counter0
    # The interesting side observation is that if the query has multiple IDX-SCANS, we will produce a
    # record for each index-scan on the child table.
    idx_scans.drop(columns=[f"counter{i}" for i in range(0, 10)] + AUX_COLUMNS, inplace=True)
    idx_scans.set_index(PER_QUERY_INDEX, inplace=True)
    roots = roots.join(idx_scans, how='left')
    roots.fillna(value={"num_defrag": 0}, inplace=True)

    sel_op = roots.query_text.str.contains("select")
    roots.loc[sel_op, "num_modify"] = roots[sel_op].num_modify + roots[sel_op].num_modify_idx

    useful_vec = roots.index.isin(idx_scans.index)
    roots.loc[useful_vec, "target"] = roots.loc[useful_vec, "target_scan"]
    roots.drop(columns=["target_scan", "num_modify_idx"], inplace=True)
    del idx_scans
    return roots


def handle_deletes(process, roots):
    # Handle the case of ModifyTableDelete.
    deletes = process[process.comment == 'ModifyTableDelete']
    deletes = deletes.reset_index(drop=True)
    # counter5 is ModifyTableDelete_num_deletes
    deletes["num_modify_del"] = deletes.counter5
    deletes["target_delete"] = deletes["target"]
    deletes.drop(columns=[f"counter{i}" for i in range(0, 10)] + AUX_COLUMNS, inplace=True)
    deletes.set_index(PER_QUERY_INDEX, inplace=True)
    roots = roots.join(deletes, how='left')
    roots.fillna(value={"num_modify_del": 0}, inplace=True)
    roots["num_modify"] = roots.num_modify + roots.num_modify_del

    useful_vec = roots.index.isin(deletes.index)
    roots.loc[useful_vec, "target"] = roots.loc[useful_vec, "target_delete"]
    roots.drop(columns=["num_modify_del", "target_delete"], inplace=True)
    del deletes
    return roots


def handle_updates(process, roots):
    # Handle the case of ModifyTableUpdate.
    updates = process[process.comment == 'ModifyTableUpdate']
    updates = updates.reset_index(drop=True)
    updates.set_index(PER_QUERY_INDEX, inplace=True)
    # counter8 is ModifyTableUpdate_num_updates.
    # counter1 is ModifyTableUpdate_num_index_updates_fired.
    # counter4 is ModifyTableUpdate_num_extends.
    # counter5 is ModifyTableUpdate_num_key_changes.
    assert np.sum(updates.counter5 == 0) == updates.shape[0]
    assert np.sum(updates.counter8 >= updates.counter1) == updates.shape[0]
    updates["num_modify_upd"] = updates.counter8
    updates["num_extend_upd"] = updates.counter4
    updates["num_hot"] = updates.counter8 - updates.counter1
    updates["target_update"] = updates["target"]
    updates.drop(columns=[f"counter{i}" for i in range(0, 10)] + AUX_COLUMNS, inplace=True)
    roots = roots.join(updates, how='left')
    roots.fillna(value={"num_modify_upd": 0, "num_extend_upd": 0, "num_hot": 0}, inplace=True)
    roots["num_modify"] = roots.num_modify + roots.num_modify_upd
    roots["num_extend"] = roots.num_extend + roots.num_extend_upd

    useful_vec = roots.index.isin(updates.index)
    roots.loc[useful_vec, "target"] = roots.loc[useful_vec, "target_update"]
    roots.drop(columns=["num_modify_upd", "num_extend_upd", "target_update"], inplace=True)
    del updates
    return roots


def analyze(input_dir, output_dir, workload_only, pg_qss_plans, query_template_map, indexoid_table_map):
    # Read in the stats and split into table and slices.
    max_txn = None
    max_time = None

    deferred = None
    CHUNK_SIZE = 8192 * 32 if workload_only else 8192 * 128
    query_id_index = ["query_id", "db_id", "pid"]

    chunk_num = 0
    it = pd.read_csv(f"{input_dir}/pg_qss_stats.csv", chunksize=CHUNK_SIZE, iterator=True)

    with tqdm() as pbar:
        chunk = it.get_chunk()
        next_chunk = it.get_chunk()
        while chunk is not None:
            if max_time is not None:
                # Assert that the data is in ascending order.
                assert np.sum(chunk.statement_timestamp < max_time) == 0
            max_time = np.max(chunk.statement_timestamp)
            # Since the timestamp is ascending, we want to process in increasing order and omit the last txn.
            max_txn = chunk.iloc[-1].txn

            # Only consider deferring if there is possibly more data.
            if next_chunk is not None:
                # Only process timestamps less than max_time. This is because there might still be
                # data in the future that we aren't yet aware of.
                process = chunk[(chunk.statement_timestamp != max_time) & (chunk.txn != max_txn)].copy()
                to_defer = chunk[(chunk.statement_timestamp == max_time) | (chunk.txn == max_txn)]
            else:
                process = chunk

            # Join with the previous block. This assumes that 1 timestamp/txn does not span a block.
            if deferred is not None and deferred.shape[0] > 0:
                process = pd.concat([process, deferred], ignore_index=True)
                deferred = to_defer
            else:
                deferred = to_defer

            # Merge the plan feature data.
            process.set_index(keys=["statement_timestamp"], drop=True, append=False, inplace=True)
            process.sort_index(axis=0, inplace=True)
            process = pd.merge_asof(process, pg_qss_plans, left_by=query_id_index, right_by=query_id_index, left_index=True, right_index=True, allow_exact_matches=True)
            process.reset_index(drop=False, inplace=True)
            process.drop(process[process.query_text.isna()].index, inplace=True)

            process["num_rel_refs"] = 0
            for tbl in TABLES:
                # Count the number of relations used in the query.
                # FIXME(MODIFY): Assume that we all modification queries impact only 1 table.
                tbl_vec = process.query_text.str.contains(tbl)
                process.loc[tbl_vec, "target"] = tbl
                process.loc[tbl_vec, "num_rel_refs"] = process.loc[tbl_vec, "num_rel_refs"] + 1

            roots = process[process.plan_node_id == -1].copy()
            if workload_only:
                for tbl in TABLES:
                    # This will be populated later with the "estimated" match count in the PK keyspace.
                    # FIXME(INDEX): We should probably separate these based on keyspace.
                    roots[f"{tbl}_pk_output"] = np.nan
                roots.drop(columns=["id", "generation"], inplace=True)
            else:
                roots.drop(columns=["id", "generation", "target"], inplace=True)

            roots.set_index(PER_QUERY_INDEX, inplace=True)
            roots.drop(columns=[f"counter{i}" for i in range(0, 10)] + ["plan_node_id", "payload"], inplace=True)

            # FIXME(ABORT): We theoretically need to handle TXN ABORT here. The difficulty is not in flagging that a query aborted,
            # but maintaining the visibility w.r.t to an aborted transaction. It is possible that under the correct visibility
            # we can model the ABORT by inserting the "inverse" modification queries. Punt for now.
            #
            # handle_abort()
            roots.drop(columns=["txn"], inplace=True)

            if not workload_only:
                roots = handle_inserts(process, roots)
                roots = handle_idxscans(process, roots, indexoid_table_map)
                roots = handle_deletes(process, roots)
                roots = handle_updates(process, roots)

            # FIXME(INDEX): We theoretically also should process index splits/extends here, however, we observe that INDEX updates are
            # not the most important. This is because under high HOT workloads -> the differential between HOT and not is reasonably
            # significant. Eventually we should have some model for % extend and % split.

            # Now featurize the "OP".
            select_vec = roots.query_text.str.contains("select")
            insert_vec = roots.query_text.str.startswith("insert")
            update_vec = roots.query_text.str.startswith("update")
            delete_vec = roots.query_text.str.startswith("delete")
            roots.loc[select_vec, "OP"] = "SELECT"
            roots.loc[insert_vec, "OP"] = "INSERT"
            roots.loc[update_vec, "OP"] = "UPDATE"
            roots.loc[delete_vec, "OP"] = "DELETE"

            # Set the modify_target by whether the query is an insert/update/delete.
            insupdel = insert_vec | update_vec | delete_vec
            roots["modify_target"] = None
            roots.loc[insupdel, "modify_target"] = roots.loc[insupdel, "target"]
            if workload_only:
                # Drop target if we only have the workload.
                roots.drop(columns=["target"], inplace=True)

            # Extract a unix_timestamp field.
            roots.reset_index(drop=False, inplace=True)
            roots["unix_timestamp"] = (roots.statement_timestamp / float(1e6)) + 86400 * (2451545 - 2440588)

            # Extract all the parameters by unquoting the outer quotes...
            splinter = roots.comment.str.split(",", expand=True)
            arg_map = {}
            for idx in splinter.columns:
                arg_map[f"arg{idx+1}"] = (splinter.loc[:, idx].str.split(" = ", expand=True)[1]).str.slice(1, -1)

            # Apply optimistic remapping of table-parameters into the actual query parameters.
            # Pray for the memory being sufficient.
            # How Elegant!

            arg_groups = []
            for qgroup in roots.groupby(by=["query_text"]):
                remap = query_template_map[qgroup[0]]
                for k, v in remap.items():
                    if not k.startswith("arg"):
                        # In this case, we don't do anything. The cross-join relationship should still
                        # be captured by the query_template_map.
                        continue

                    try:
                        qgroup[1][v] = arg_map[k][qgroup[1].index].astype(float)
                    except:
                        qgroup[1][v] = arg_map[k][qgroup[1].index]
                arg_groups.append(qgroup[1])

            roots = pd.concat(arg_groups, ignore_index=True)
            roots.sort_values(by=["statement_timestamp"], inplace=True, ignore_index=True)
            roots.drop(columns=["comment"], inplace=True)

            roots.reset_index(drop=True).to_feather(f"{output_dir}/analysis/chunk_{chunk_num}.feather")
            chunk_num += 1

            chunk = next_chunk
            try:
                next_chunk = it.get_chunk()
            except StopIteration:
                next_chunk = None
            pbar.update()


def analyze_workload(benchmark, input_dir, output_dir, workload_only, psycopg2_conn):
    global TABLES
    TABLES = BENCHDB_TO_TABLES[benchmark]

    # If we only have the workload, then we don't have VACUUM and window timestamps.
    window_index_map = {}
    if not workload_only:
        pg_stat_user_tables = pd.read_csv(f"{input_dir}/pg_stat_user_tables.csv")
        pg_stat_user_tables = pg_stat_user_tables[~pg_stat_user_tables.last_autovacuum.isnull()]
        pg_stat_user_tables["autovacuum_unix_timestamp"] = pd.to_datetime(pg_stat_user_tables.last_autovacuum).map(pd.Timestamp.timestamp)
        for tbl in TABLES:
            sample = pd.read_csv(f"{input_dir}/{tbl}.csv")
            sample["true_window_index"] = sample.index
            sample["time"] = (sample.time / float(1e6))
            sample = sample[["time", "true_window_index"]]

            # Inject a window_index that is -1
            # window_index = -1 are tuples that get discarded! woo-hoo.

            substat = pg_stat_user_tables[pg_stat_user_tables.relname == tbl]
            autovacuum_times = [v[0] for v in substat.groupby(by=["autovacuum_unix_timestamp"])]
            if len(autovacuum_times) > 0:
                wipe_frame = pd.DataFrame([{"time": autovac, "true_window_index": -1} for autovac in autovacuum_times])
                sample = pd.concat([sample, wipe_frame], ignore_index=True)
                sample.sort_values(by=["time"], ignore_index=True, inplace=True)
            sample["window_index"] = sample.index
            sample.set_index(["time"], inplace=True)
            window_index_map[tbl] = sample

    # Here we construct all INDEX columns that are possibly of value.
    # This is done by extracting all INDEXed key columns and then augmenting on IDX-JOINs.
    table_index_map = {t: set() for t in TABLES}
    # table_attr_map defines all attributes that might exist per table.
    # We hereby assume that attribute names are unique.
    table_attr_map = {t: [] for t in TABLES}
    attr_table_map = {}
    # Defines the keyspace for a relation. This is taken as the INDEX space OR the PK space.
    # All tuple inserts/deletes are done relative to the PK space. INDEX space is used
    # for looking at data distribution w.r.t to index clustering.
    table_keyspace_map = {t: {} for t in TABLES}
    # Process pg_index and pg_class from files to build the indexoid_table_map.
    indexoid_table_map = {}

    if workload_only:
        assert psycopg2_conn is not None
        with psycopg.connect(psycopg2_conn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                # Execute the query to get a mapping from attribute -> table.
                result = cursor.execute(QUERY_TABLE_ATTRIBUTE_COLUMNS)
                [table_attr_map[tup[0]].append(tup[1]) for tup in result if tup[0] in table_attr_map]

                # Execute to build index key columns.
                result = cursor.execute(QUERY_TABLE_INDEX_COLUMNS)
                for tup in result:
                    if tup[0] in table_index_map:
                        table_index_map[tup[0]] = table_index_map[tup[0]].union(tup[4].split(","))

                        table_keyspace_map[tup[0]][tup[1]] = tup[4].split(",")
                        if tup[3]:
                            table_keyspace_map[tup[0]][tup[0]] = tup[4].split(",")

        for tbl in TABLES:
            for attr in table_attr_map[tbl]:
                assert attr not in attr_table_map
                attr_table_map[attr] = tbl
    else:
        # FIXME(NON_SCHEMA): We assume that there aren't useful schema changes to be identified
        # by assuming that sample at t=0 for the schema is the same at any point in the data file.
        pg_index = pd.read_csv(f"{input_dir}/pg_index.csv")
        pg_index = pg_index[pg_index.time == pg_index.iloc[-1].time]

        pg_class = pd.read_csv(f"{input_dir}/pg_class.csv")
        pg_class = pg_class[pg_class.time == pg_class.iloc[-1].time]

        pg_attribute = pd.read_csv(f"{input_dir}/pg_attribute.csv")
        pg_attribute = pg_attribute[pg_attribute.time == pg_attribute.iloc[-1].time]
        pg_attribute.set_index(keys=["attrelid"], inplace=True)

        pg_index.set_index(keys=["indrelid"], inplace=True)
        pg_class.set_index(keys=["oid"], inplace=True)
        pg_index = pg_index.join(pg_class, how="inner", rsuffix="_class")
        pg_index.reset_index(drop=False, inplace=True)
        pg_index["indrelid"] = pg_index["index"]
        pg_index.drop(columns=["index"], inplace=True)

        pg_index.set_index(keys=["indexrelid"], inplace=True)
        pg_index = pg_index.join(pg_class, how="inner", rsuffix="_index")
        pg_index.reset_index(drop=False, inplace=True)
        pg_index["indexrelid"] = pg_index["index"]
        pg_index.drop(columns=["index"], inplace=True)

        pg_attribute = pg_attribute.join(pg_class, how="inner", rsuffix="_class")
        pg_attribute.reset_index(drop=False, inplace=True)
        for pgatt in pg_attribute.groupby(by=["relname"]):
            # Construct the table_attr_map.
            if pgatt[0] in table_attr_map:
                table_attr_map[pgatt[0]] = pgatt[1][pgatt[1].attnum >= 1].attname.values

        for tbl in TABLES:
            for attr in table_attr_map[tbl]:
                assert attr not in attr_table_map
                attr_table_map[attr] = tbl

        for tup in pg_index.itertuples():
            # Construct the indexoid_table_map.
            indexoid_table_map[tup.indexrelid] = tup.relname

            attnums = tup.indkey.split(" ")
            table_keyspace_map[tup.relname][tup.relname_index] = []
            for attnum in attnums:
                df = (pg_attribute[(pg_attribute.relname == tup.relname) & (pg_attribute.attnum == int(attnum))])
                assert len(df) == 1

                att = df.iloc[0]
                if tup.relname in table_index_map:
                    table_index_map[tup.relname].add(att.attname)
                    table_keyspace_map[tup.relname][tup.relname_index].append(att.attname)

            if tup.indisprimary:
                # Handle the primary key for the table keyspace.
                table_keyspace_map[tup.relname][tup.relname] = table_keyspace_map[tup.relname][tup.relname_index]

    # Process all the plans to get the query text.
    pg_qss_plans = pd.read_csv(f"{input_dir}/pg_qss_plans.csv")
    pg_qss_plans["id"] = pg_qss_plans.index
    pg_qss_plans["query_text"] = ""
    for plan in pg_qss_plans.itertuples():
        feature = json.loads(plan.features)
        query_text = feature[0]["query_text"].lower().rstrip()

        for query in QUERY_CONTENT_BLOCK:
            if query_text is not None and query in query_text:
                query_text = None
        pg_qss_plans.at[plan.Index, "query_text"] = query_text
    pg_qss_plans.drop(labels=["features"], axis=1, inplace=True)
    pg_qss_plans.drop(pg_qss_plans[pg_qss_plans.query_text.isna()].index, inplace=True)
    pg_qss_plans.reset_index(drop=True, inplace=True)
    pg_qss_plans.set_index(keys=["statement_timestamp"], drop=True, append=False, inplace=True)
    pg_qss_plans.sort_index(axis=0, inplace=True)

    # Process each query and extract useful parameters w.r.t to INDEX keyspaces.
    query_template_map = {}
    for plan_group in pg_qss_plans.groupby(by=["query_text"]):
        query_template_map[plan_group[0]] = {}
        root = pglast.Node(pglast.parse_sql(plan_group[0]))
        for node in root.traverse():
            if isinstance(node, pglast.node.Node):
                if isinstance(node.ast_node, pglast.ast.InsertStmt):
                    # Specific case of INSERT INTO [tbl] (cols) VALUES ()
                    values = node.ast_node.selectStmt.valuesLists[0]
                    for i, target in enumerate(node.ast_node.cols):
                        column = target.name
                        param = values[i].number
                        query_template_map[plan_group[0]][f"arg{param}"] = column
                elif isinstance(node.ast_node, pglast.ast.A_Expr):
                    # It can be of the form [column] [OP] [param]
                    # Or it can be of the form [param] [OP] [column]
                    column = None
                    param = None
                    ast = node.ast_node
                    if (isinstance(ast.lexpr, pglast.ast.ColumnRef) and isinstance(ast.rexpr, pglast.ast.ParamRef)):
                        column = ast.lexpr.fields[0].val
                        param = ast.rexpr.number
                    elif (isinstance(ast.lexpr, pglast.ast.ParamRef) and isinstance(ast.rexpr, pglast.ast.ColumnRef)):
                        column = ast.rexpr.fields[0].val
                        param = ast.lexpr.number

                    if column is not None:
                        operator = node.ast_node.name[0].val
                        # Range clause.
                        if operator == "<":
                            column = column + "_high"
                        elif operator == ">=":
                            column = column + "_loweq"
                        # This is an useful index column.
                        assert param is not None
                        query_template_map[plan_group[0]][f"arg{param}"] = column

                    if (isinstance(ast.lexpr, pglast.ast.ColumnRef) and isinstance(ast.rexpr, pglast.ast.ColumnRef)):
                        column = ast.lexpr.fields[0].val
                        param = ast.rexpr.fields[0].val
                        # This is the case where there is [a] column = [b] column.
                        query_template_map[plan_group[0]][param] = column
                        query_template_map[plan_group[0]][column] = param

    # Purge query template map of unnecessary keys (consult table_index_map and other in query template map).
    # After the purge, we have all keys that are used or referenced through joins.
    query_templates = query_template_map.keys()
    # We don't want to prune the cross-join keys since we need that to determine table_attr_map.
    for key in query_templates:
        query_template_map[key] = {k:v for k,v in query_template_map[key].items()
                                        if (any([v in table_index_map[t] for t in TABLES]) or any([k in query_template_map[q] for q in query_template_map]))}

    # We need to define global keys of interest here.
    for tbl in TABLES:
        old_attr_list = table_attr_map[tbl]
        new_attr_list = [a for a in old_attr_list if any([a in table_index_map[t] for t in TABLES]) or any([a in query_template_map[q] for q in query_template_map])]
        table_attr_map[tbl] = new_attr_list

    # Write useful metadata information to keyspaces.pickle
    Path(f"{output_dir}/analysis/").mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/analysis/keyspaces.pickle", "wb") as f:
        pickle.dump(table_attr_map, f)
        pickle.dump(attr_table_map, f)
        pickle.dump(table_keyspace_map, f)
        pickle.dump(query_template_map, f)
        pickle.dump(window_index_map, f)

    analyze(input_dir, output_dir, workload_only, pg_qss_plans, query_template_map, indexoid_table_map)


class AnalyzeWorkloadCLI(cli.Application):
    benchmark = cli.SwitchAttr(
        "--benchmark",
        str,
        mandatory=True,
        help="Benchmark that should be analyzed.",
    )

    dir_workload_input = cli.SwitchAttr(
        "--dir-workload-input",
        str,
        mandatory=True,
        help="Path to the folder containing the workload input.",
    )

    dir_workload_output = cli.SwitchAttr(
        "--dir-workload-output",
        str,
        mandatory=True,
        help="Path to the folder containing the output of the analyzed workload.",
    )

    workload_only = cli.SwitchAttr(
        "--workload-only",
        str,
        help="Whether the input contains only the workload stream.",
    )

    psycopg2_conn = cli.SwitchAttr(
        "--psycopg2-conn",
        str,
        help="Psycopg2 connection that should be used.",
    )

    def main(self):
        b_parts = self.benchmark.split(",")
        input_parts = self.dir_workload_input.split(",")
        output_parts = self.dir_workload_output.split(",")
        for i in range(len(output_parts)):
            logger.info("Processing %s -> %s (%s, %s)", input_parts[i], output_parts[i], b_parts[i], self.workload_only)
            analyze_workload(b_parts[i], input_parts[i], output_parts[i], (self.workload_only == "True"), self.psycopg2_conn)


if __name__ == "__main__":
    AnalyzeWorkloadCLI.run()
