import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from psycopg.rows import dict_row
from pandas.api import types as pd_types

from behavior import OperatingUnit
from behavior.model_workload.utils import OpType


TABLE_EXEC_FEATURES = [
    ("query_order", "bigint"),
    ("statement_timestamp", "bigint"),
    ("unix_timestamp", "float8"),
    ("optype", "int"),
    ("txn", "int"),
    ("target", "text"),
    ("num_modify_tuples", "int"),
    ("num_select_tuples", "int"),
    ("num_extend", "int"),
    ("num_hot", "int"),
    ("num_defrag", "int")
]


# The use of MAX() is really funky but that's because postgres doesn't have an understanding
# of a first/any aggregate. In reality, the numbers computed are static across the window
# because of the inner query. So we can just take any value, as far as i know.
# (famous last words)
TABLE_EXEC_FEATURES_QUERY = """
	SELECT
		s.query_order,
		s.statement_timestamp,
		s.unix_timestamp,
		s.optype,
		s.txn,
		s.target,
		MAX(s.num_modify_tuples) as num_modify_tuples,
		MAX(s.num_select_tuples) as num_select_tuples,
		MAX(s.num_extend) as num_extend,
		MAX(s.num_hot) as num_hot,
		MAX(s.num_defrag) as num_defrag
	FROM (
		SELECT
			a.query_order,
			a.query_id,
			a.statement_timestamp,
			a.unix_timestamp,
			a.optype,
			a.txn,
			a.comment,
			COALESCE(b.target_idx_scan_table, b.target) AS target,

			SUM(CASE b.comment
			    WHEN 'ModifyTableInsert' THEN 1
			    WHEN 'ModifyTableUpdate' THEN b.counter8
			    WHEN 'ModifyTableDelete' THEN b.counter5
			    ELSE 0 END) OVER w AS num_modify_tuples,

			SUM(CASE b.comment
			    WHEN 'IndexScan' THEN b.counter0
			    WHEN 'IndexOnlyScan' THEN b.counter0
			    WHEN 'SeqScan' THEN b.counter0
			    ELSE 0 END) OVER w AS num_select_tuples,

			SUM(CASE b.comment
			    WHEN 'ModifyTableInsert' THEN b.counter4
			    WHEN 'ModifyTableUpdate' THEN b.counter4
			    ELSE 0 END) OVER w AS num_extend,

			SUM(CASE b.comment
			    WHEN 'ModifyTableUpdate' THEN b.counter8 - b.counter1
			    ELSE 0 END) OVER w AS num_hot,

			SUM(CASE b.comment
			    WHEN 'IndexScan' THEN b.counter3
			    WHEN 'IndexOnlyScan' THEN b.counter3
			    WHEN 'SeqScan' THEN b.counter1
			    ELSE 0 END) OVER w AS num_defrag

        FROM {work_prefix}_mw_queries_args a
        LEFT JOIN {work_prefix}_mw_queries b ON a.query_order = b.query_order AND b.plan_node_id != -1
        WHERE a.target = '{target}'
        WINDOW w AS (PARTITION BY a.query_order, b.target_idx_scan_table)
    ) s
    WHERE s.comment != 'ModifyTableIndexInsert' AND position(',' in s.target) = 0
    GROUP BY s.query_order, s.statement_timestamp, s.unix_timestamp, s.optype, s.txn, s.target;
"""


def build_table_exec(logger, connection, work_prefix, tables):
    logger.info("Building execution statistics.")
    with connection.transaction():
        sql = f"CREATE UNLOGGED TABLE {work_prefix}_mw_stats ("
        sql += ",".join([f"{k} {v}" for k, v in TABLE_EXEC_FEATURES])
        sql += ") WITH (autovacuum_enabled = OFF)"
        connection.execute(sql)

        for t in tables:
            sql = f"INSERT INTO {work_prefix}_mw_stats " + TABLE_EXEC_FEATURES_QUERY.format(work_prefix=work_prefix, target=t)
            logger.info("%s", sql)
            c = connection.execute(sql)
            logger.info("Finished executing: %s", c.rowcount)


def __build_dist_query(work_prefix, tbl, att_keys, query_orders, buckets, compute_data, compute_optype):
    sqls = []
    query_orders = [str(l) for l in query_orders]

    # Extract a nesting of window index to query orders.
    # [query_order[0], query_order[1]] is window 0; everything that happens before is folded in.
    window_cte = """
        CREATE UNLOGGED TABLE w WITH (autovacuum_enabled=OFF) AS
        (SELECT nr-1 AS window_index, elem::integer as point FROM UNNEST(ARRAY[{query_orders}]) WITH ORDINALITY AS u(elem, nr))
    """.format(query_orders=",".join(query_orders))
    sqls.append(window_cte)
    sqls.append("CREATE INDEX w_idx on w (window_index)")

    # data statistics CTE. Compute min/max based on the visibility at each window start point.
    extracts = [f"s{2*i}.{k} as min_{k}, s{2*i+1}.{k} as max_{k}\n" for i, k in enumerate(att_keys)]
    extract_tbls = ["""
        LATERAL (
            SELECT {k} FROM {work_prefix}_{tbl} a
            WHERE a.insert_version <= w.point AND (a.delete_version IS NULL or a.delete_version >= w.point)
            ORDER BY {k} ASC LIMIT 1) s{min_s},
        LATERAL(
            SELECT {k} FROM {work_prefix}_{tbl} a
            WHERE a.insert_version <= w.point AND (a.delete_version IS NULL or a.delete_version >= w.point)
            ORDER BY {k} DESC LIMIT 1) s{max_s}
        """.format(k=k, work_prefix=work_prefix, tbl=tbl, min_s=2*i, max_s = 2*i+1) for i, k in enumerate(att_keys)]

    base_data_summary_cte = """
        CREATE UNLOGGED TABLE data_state WITH (autovacuum_enabled=OFF) AS (
        SELECT w.window_index,
               {extract_columns}
               FROM w, {extract_tbls}
        )
    """.format(extract_columns=",".join(extracts), extract_tbls=",".join(extract_tbls))
    sqls.append(base_data_summary_cte)
    sqls.append("CREATE INDEX data_state_idx ON data_state (window_index)")

    # Try and pre-warm since these are big indexes.
    sqls.extend([f"SELECT pg_prewarm('{work_prefix}_{tbl}_hits_{i}')" for i in range(len(att_keys))])

    # hits statistics CTE.
    diff_keys = [f"LEAST(h.min_{k}, d.min_{k}) AS min_{k},\nGREATEST(h.max_{k}, d.max_{k}) AS max_{k}\n" for k in att_keys]
    extract_tbls = ["""
        LATERAL (
            SELECT {k} FROM {work_prefix}_{tbl}_hits a
            WHERE a.query_order >= w2.point AND a.query_order <= w.point
            ORDER BY {k} ASC LIMIT 1) s{min_s},
        LATERAL(
            SELECT {k} FROM {work_prefix}_{tbl}_hits a
            WHERE a.query_order >= w2.point AND a.query_order <= w.point
            ORDER BY {k} DESC LIMIT 1) s{max_s}
        """.format(k=k, work_prefix=work_prefix, tbl=tbl, min_s=2*i, max_s = 2*i+1) for i, k in enumerate(att_keys)]
    hits_data_cte = """
        CREATE UNLOGGED TABLE hits_state WITH (autovacuum_enabled=OFF) AS
        (
            SELECT h.window_index,
            h.point,
            {diff_keys}
            FROM (
                SELECT w.window_index - 1 as window_index,
                w.point,
                {extract_columns}
                FROM w, w as w2, {extract_tbls}
                WHERE w.window_index > 0 AND w.window_index = w2.window_index + 1
            ) h, data_state d WHERE h.window_index = d.window_index
        )
    """.format(diff_keys=",".join(diff_keys), extract_columns=",".join(extracts), extract_tbls=",".join(extract_tbls))
    sqls.append(hits_data_cte)
    sqls.append("CREATE INDEX hits_state_idx ON hits_state (window_index)")

    # For hits, We remove the first query_order since we want to use the fact
    # that width_bucket() less than the first element = 0. that way we are adjusting over
    # the correct target window; and so we compute hits_state such that min/max is with
    # regards to what happens in the interval.
    clauses = [(f"width_bucket(CASE WHEN h.min_{k} = h.max_{k} THEN 1.0 ELSE (a.{k} - h.min_{k})::float / (h.max_{k} - h.min_{k}) END, 0.0, 1.0, {buckets-1})-1", k) for k in att_keys]
    preclude = "h.window_index " if compute_data or not compute_optype else "h.window_index, a.optype "
    f = "a.insert_version <= h.point AND (a.delete_version IS NULL OR a.delete_version >= h.point)" if compute_data \
        else "width_bucket(a.query_order, ARRAY[{qo}]) = h.window_index".format(qo=",".join(query_orders[1:]))
    dist_sql = """
        CREATE UNLOGGED TABLE data WITH (autovacuum_enabled=OFF) AS (
            SELECT {preclude},
                   {clauses},
                   count(1) as freq
              FROM {work_prefix}_{tbl} a, hits_state h
             WHERE {filter}
             GROUP BY GROUPING SETS ({groups})
        )
    """.format(
        preclude=preclude,
        clauses=",".join([f"{c[0]} as {c[1]}" for c in clauses]),
        work_prefix=work_prefix,
        tbl=tbl if compute_data else tbl + "_hits",
        filter=f,
        groups=",".join([f"({preclude}, {c[0]})\n" for c in clauses])
    )
    sqls.append(dist_sql)
    sqls.append("SELECT * FROM data")
    return sqls, ["w", "data_state", "hits_state", "data"]


def __execute_dist_query(logger, cursor, work_prefix, tbl, att_keys, query_orders, buckets, compute_data, compute_optype=False):
    # Generate all the SQLs we need to execute.
    sqls, purge_tbls = __build_dist_query(work_prefix, tbl, att_keys, query_orders, buckets, compute_data, compute_optype)

    with open("/tmp/query", "w") as f:
        for s in sqls:
            f.write(s)

    # Now execute the SQLs; take the last result.
    for i, s in enumerate(sqls):
        logger.info("Executing SQL: [%s/%s]", i+1, len(sqls))
        result = cursor.execute(s)

    output_tuples = []
    rows = [r for r in result]
    columns = ["window_index"] + (["optype"] if not compute_data and compute_optype else []) + att_keys + ["freq_count"]
    df = pd.DataFrame(rows, columns=columns)

    gb = ["window_index"] if compute_data or not compute_optype else ["window_index", "optype"]
    for att in att_keys:
        subframe = df[~df[att].isna()]
        for grp, frame in subframe.groupby(by=gb):
            total = frame.freq_count.sum()
            bucket_list = np.zeros(buckets)
            np.put(bucket_list, frame[att].values.astype(int), frame.freq_count / total)
            if compute_data:
                output_tuples.append([grp, "data", att, bucket_list.tolist()])
            elif not compute_optype:
                output_tuples.append([grp, "all", att, bucket_list.tolist()])
            else:
                opname = OpType(grp[1])
                output_tuples.append([grp[0], opname.name, att, bucket_list.tolist()])

    columns = ["window_index", "optype", "att_name", "key_dist"]
    for tbl in purge_tbls:
        cursor.execute(f"DROP TABLE {tbl}")
    return pd.DataFrame(output_tuples, columns=columns)


def construct_keyspaces(logger, connection, work_prefix, tbls, table_attr_map, window_index_map, buckets, gen_data=True, gen_op=True, callback_fn=None):
    datatypes = {}
    with connection.transaction():
        result = connection.execute("SELECT table_name, column_name, data_type FROM information_schema.columns")
        for record in result:
            tbl, att, dt = record[0], record[1], record[2]
            if not tbl in datatypes:
                datatypes[tbl] = {}
            datatypes[tbl][att] = dt

    bucket_ks = {}
    with connection.cursor() as cursor:
        for tbl in tbls:
            if isinstance(window_index_map[tbl], range) or isinstance(window_index_map[tbl], list):
                query_orders = window_index_map[tbl]
            else:
                # Assume it is a data frame then.
                query_orders = window_index_map[tbl].query_order.values

            attrs = []
            for a in table_attr_map[tbl]:
                mod_tbl = f"{work_prefix}_{tbl}"
                if mod_tbl in datatypes and a in datatypes[mod_tbl]:
                    t = datatypes[mod_tbl][a]
                    if not (t == "text" or "character" in t):
                        attrs.append(a)

            if len(attrs) == 0:
                # There aren't any attributes of interest.
                logger.info("Skipping querying keyspace distribution for %s", tbl)
                continue

            if gen_data:
                logger.info("Querying keyspace distribution for raw data (%s): %s (%s)", attrs, tbl, datetime.now())
                data_ks = __execute_dist_query(logger, cursor, work_prefix, tbl, attrs, query_orders, buckets, True, compute_optype=False)

            logger.info("Querying keyspace distribution from access: %s (%s)", tbl, datetime.now())
            op_ks = __execute_dist_query(logger, cursor, work_prefix, tbl, attrs, query_orders, buckets, False, compute_optype=gen_op)

            if gen_data:
                # Concat the data segment if we need to.
                df = pd.concat([data_ks, op_ks], ignore_index=True)
            else:
                df = op_ks

            if callback_fn is not None:
                df = callback_fn(tbl, df)
                if df is not None:
                    bucket_ks[tbl] = df
            else:
                bucket_ks[tbl] = df
    return bucket_ks


def construct_query_states(logger, connection, work_prefix, tbls, window_index_map, buckets):
    tbl_ks = {}
    agg_stats = [
        "num_modify_tuples",
        "num_select_tuples",
        "num_extend",
        "num_hot",
        "num_defrag",
    ]

    ops = [
        ("num_insert", OpType.INSERT.value),
        ("num_update", OpType.UPDATE.value),
        ("num_delete", OpType.DELETE.value),
    ]

    qs = [
        ("num_select_queries", OpType.SELECT.value),
        ("num_insert_queries", OpType.INSERT.value),
        ("num_update_queries", OpType.UPDATE.value),
        ("num_delete_queries", OpType.DELETE.value),
    ]

    with connection.cursor() as cursor:
        for tbl in tbls:
            logger.info("Computing data keys for %s", tbl)

            # We have to truncate the first value off. This is because width_bucket() has the dynamic that anything less
            # than the 1st element returns index 0 (which is marked by the first entry of window_index_map[tbl].
            #
            # The 0th window of window_index_map spans all queries between window_index_map[tbl].time[0] and .time[1].
            # Which means we need width_bucket() to return 0 for QOs betweeen time[0] and time[1].
            query_orders = [str(i) for i in window_index_map[tbl].query_order.values[1:]]
            sql = "SELECT width_bucket(query_order, ARRAY[" + ",".join(query_orders) + "]) as window_index, "
            sql += ",".join([f"SUM({k})" for k in agg_stats]) + ", "
            sql += ",".join([f"SUM(CASE optype WHEN {v} THEN num_modify_tuples ELSE 0 END) AS {f}" for f, v in ops]) + ", "
            sql += ",".join([f"SUM(CASE optype WHEN {v} THEN 1 ELSE 0 END) AS {f}" for f, v in qs]) + " "
            sql += f"FROM {work_prefix}_mw_stats "
            sql += f"WHERE target = '{tbl}' "
            sql += "GROUP BY window_index"

            records = []
            result = cursor.execute(sql)
            for r in result:
                # This actually belongs to a window that we can use.
                if window_index_map[tbl].iloc[r[0]].true_window_index != -1:
                    records.append(list(r))

            tbl_ks[tbl] = pd.DataFrame(records, columns=["window_index"] + agg_stats + [f for f, _ in ops] + [f for f, _ in qs])

    return tbl_ks
