import gc
import re
import json
from tqdm import tqdm
from pathlib import Path
from enum import Enum, auto, unique
import pandas as pd
import numpy as np


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
    query_stats = pg_qss_stats[(pg_qss_stats.plan_node_id == -1) & (pg_qss_stats.query_id != 0)]
    # counter0 is 1 if the query has successfully executed.
    query_stats = query_stats[query_stats.counter0 == 1]
    query_stats.drop(columns=[f"counter{i}" for i in range(0, 10)] + ["plan_node_id"], inplace=True, errors='raise')
    query_stats["params"] = query_stats.comment.fillna('')
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
    query_stats.drop(labels=["params", "id", "generation", "comment", "txn"], axis=1, inplace=True)

    # Sort by statement_timestamp so we are processing the query stream in serial.
    query_stats.set_index(keys=["statement_timestamp"], drop=True, append=False, inplace=True)
    query_stats.sort_index(axis=0, inplace=True)
    # We don't actually want the statement_timestamp since that is "real" time but we do want the queries
    # in the correct execution order.
    query_stats["order"] = np.arange(len(query_stats))
    query_stats.reset_index(drop=False, inplace=True)
    return query_stats
