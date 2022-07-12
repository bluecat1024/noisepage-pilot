from __future__ import annotations
from enum import Enum

# A mapping from supported benchmark to tables used in the benchmark. This mapping is primarily
# used by doit behavior_pg_analyze_benchmark.
BENCHDB_TO_TABLES = {
    "tpcc": [
        "warehouse",
        "district",
        "customer",
        "item",
        "stock",
        "oorder",
        "history",
        "order_line",
        "new_order",
    ],
    "tatp": [
        "subscriber",
        "special_facility",
        "access_info",
        "call_forwarding",
    ],
    "tpch": [
        "region",
        "nation",
        "customer",
        "supplier",
        "part",
        "orders",
        "partsupp",
        "lineitem",
    ],
    "wikipedia": [
        "useracct",
        "watchlist",
        "ipblocks",
        "logging",
        "user_groups",
        "recentchanges",
        "page",
        "revision",
        "page_restrictions",
        "text",
    ],
    "voter": [
        "contestants",
        "votes",
        "area_code_state",
    ],
    "twitter": ["user_profiles", "tweets", "follows", "added_tweets", "followers"],
    "smallbank": ["accounts", "checking", "savings"],
    "sibench": ["sitest"],
    "seats": [
        "country",
        "airline",
        "airport",
        "customer",
        "flight",
        "airport_distance",
        "frequent_flyer",
        "reservation",
        "config_profile",
        "config_histograms",
    ],
    "resourcestresser": ["iotable", "cputable", "iotablesmallrow", "locktable"],
    "noop": ["fake"],
    "epinions": ["item", "review", "useracct", "trust", "review_rating"],
    "auctionmark": [
        "region",
        "useracct",
        "category",
        "config_profile",
        "global_attribute_group",
        "item",
        "item_comment",
        "useracct_feedback",
        "useracct_attributes",
        "item_bid",
        "useracct_watch",
        "global_attribute_value",
        "item_attribute",
        "item_image",
        "item_max_bid",
        "item_purchase",
        "useracct_item",
    ],
    "ycsb": ["usertable"],
}


# This list must be kept up to date with the OU definitions in cmu-db/postgres.
class OperatingUnit(Enum):
    Agg = 0
    Append = 1
    BitmapAnd = 2
    BitmapHeapScan = 3
    BitmapIndexScan = 4
    BitmapOr = 5
    CteScan = 6
    CustomScan = 7
    ForeignScan = 8
    FunctionScan = 9
    Gather = 10
    GatherMerge = 11
    Group = 12
    Hash = 13
    HashJoinImpl = 14
    IncrementalSort = 15
    IndexOnlyScan = 16
    IndexScan = 17
    Limit = 18
    LockRows = 19
    Material = 20
    Memoize = 21
    MergeAppend = 22
    MergeJoin = 23
    ModifyTableInsert = 24
    ModifyTableUpdate = 25
    ModifyTableDelete = 26
    ModifyTableIndexInsert = 27
    NamedTuplestoreScan = 28
    NestLoop = 29
    ProjectSet = 30
    RecursiveUnion = 31
    Result = 32
    SampleScan = 33
    SeqScan = 34
    SetOp = 35
    Sort = 36
    SubPlan = 37
    SubqueryScan = 38
    TableFuncScan = 39
    TidScan = 40
    TidRangeScan = 41
    Unique = 42
    ValuesScan = 43
    WindowAgg =  44
    WorkTableScan = 45
    DestReceiverRemote = 46


class Targets(Enum):
    ELAPSED_US = "elapsed_us"


TARGET_COLUMNS = [n.value for n in Targets]


"""
A dictionary of derived features. The name is the name of the derived feature that follows the format
[OU]_[feature]. OU is guaranteed to be an unique identifier. The value of the derived feature is the
column in pg_qss_stats that should be remapped.
"""
DERIVED_FEATURES_MAP = {
    "DestReceiverRemote_num_output": "counter0",
    "IndexOnlyScan_num_iterator_used": "counter0",
    "IndexOnlyScan_num_heap_fetches": "counter1",
    "IndexScan_num_iterator_used": "counter0",
    "IndexScan_num_heap_fetches": "counter1",
    "IndexScan_num_outer_loops": "counter2",
    "IndexScan_num_defrag": "counter3",
    "ModifyTableInsert_num_br_ir_as_triggers_fired": "counter0",
    "ModifyTableInsert_num_spec_insert": "counter1",
    "ModifyTableInsert_num_tuple_toast": "counter2",
    "ModifyTableInsert_num_fsm_checks": "counter3",
    "ModifyTableInsert_num_extends": "counter4",
    "ModifyTableIndexInsert_indexid": "payload",
    "ModifyTableIndexInsert_num_fsm_checks": "counter0",
    "ModifyTableIndexInsert_num_extends": "counter1",
    "ModifyTableIndexInsert_num_splits": "counter2",
    "ModifyTableIndexInsert_num_finish_splits": "counter3",
    "ModifyTableUpdate_num_br_ir_as_triggers_epq_fired": "counter0",
    "ModifyTableUpdate_num_index_updates_fired": "counter1",
    "ModifyTableUpdate_num_tuple_toast": "counter2",
    "ModifyTableUpdate_num_fsm_checks": "counter3",
    "ModifyTableUpdate_num_extends": "counter4",
    "ModifyTableUpdate_num_key_changes": "counter5",
    "ModifyTableUpdate_num_acquire_tuplock": "counter6",
    "ModifyTableUpdate_num_lock_wait_members": "counter7",
    "ModifyTableUpdate_num_updates": "counter8",
    "ModifyTableUpdate_num_aborts": "counter9",
    "ModifyTableDelete_num_br_ir_as_triggers_fired": "counter0",
    "ModifyTableDelete_num_recheck_quals": "counter1",
    "ModifyTableDelete_num_tuple_returns": "counter2",
    "ModifyTableDelete_num_acquire_tuplock": "counter3",
    "ModifyTableDelete_num_lock_wait_members": "counter4",
    "LockRows_num_marks": "counter0",
    "LockRows_num_invoke_epq": "counter1",
    "LockRows_num_update_find_latest": "counter2",
    "LockRows_num_check_tuplock_state": "counter3",
    "LockRows_num_acquire_tuplock": "counter4",
    "LockRows_num_lock_wait_members": "counter5",
    "LockRows_num_heap_lock_updated_tuple_rec_tuples": "counter6",
    "LockRows_num_heap_lock_updated_tuple_rec_wait_members": "counter7",
    "LockRows_num_non_lock_wait_block_policy": "counter8",
    "LockRows_num_aborts": "counter9",
    "Agg_num_input_rows": "counter0",
    "NestLoop_num_outer_rows": "counter0",
    "NestLoop_num_inner_rows_cumulative": "counter1",
    "BitmapIndexScan_num_tids_found": "counter0",
    "BitmapHeapScan_num_blocks_fetch": "counter0",
    "BitmapHeapScan_num_empty_tuples": "counter1",
    "BitmapHeapScan_num_tuples_fetch": "counter2",
    "BitmapHeapScan_num_blocks_prefetch": "counter3",
}
