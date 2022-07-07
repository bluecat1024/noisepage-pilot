from __future__ import annotations
from behavior import OperatingUnit, TARGET_COLUMNS

PLAN_INDEPENDENT_ID = -3

# This defines the keys to use for a pandas dataframe to group by unique query invocation.
# (query_id, db_id, statement_timestamp, pid) identify a unique query invocation.
UNIQUE_QUERY_ID_INDEX = [
    "query_id",
    "db_id",
    "statement_timestamp",
    "pid"
]

#################################
# extract_qss definitions
#################################

QSS_MERGE_PLAN_KEY = UNIQUE_QUERY_ID_INDEX + ["plan_node_id"]

QSS_PLANS_IGNORE_NODE_FEATURES = [
    "plan_node_id",
    "left_child_node_id",
    "right_child_node_id",
    "node",
    "node_type",
    "query_text"
]

# The list of Operating Units that have corresponding query state store statistics
# that need to be merged. This also is the list of OUs that need to have columns
# renamed from PG opaque columns.
QSS_STATS_OUS = [
    OperatingUnit.IndexScan,
    OperatingUnit.IndexOnlyScan,
    OperatingUnit.ModifyTableInsert,
    OperatingUnit.ModifyTableUpdate,
    OperatingUnit.ModifyTableDelete,
    OperatingUnit.ModifyTableIndexInsert,
    OperatingUnit.LockRows,
    OperatingUnit.Agg,
    OperatingUnit.NestLoop,
    OperatingUnit.AfterQueryTrigger,
]

#################################
# data_diff definitions
#################################

# List of OUs that are blocked from being differenced. Blocked OUs are not copied to output.
DIFF_BLOCKED_OUS = [
    # These OUs are blocked because we currently can't difference parallel OUs.
    # By extension, all data associated with parallel queries will be dropped.
    # TODO(wz2): Support parallel OUs.
    OperatingUnit.Gather,
    OperatingUnit.GatherMerge,
]

# List of OUs that skip differencing process. These OUs are directly copied to the output.
# These OUs are most likely OUs that are traced separately.
DIFF_SKIP_OUS = [
    OperatingUnit.DestReceiverRemote,
    OperatingUnit.ModifyTableIndexInsert,
    OperatingUnit.AfterQueryTrigger,
]


# OUs read from CSV have distinct schemas since each OU contains different features that are extracted.
# `DIFFERENCING_SCHEMA` defines the minimal set of features common to each OU that are required to
# perform differencing correctly.
#
# If *you* ever consider adding to this list, please consider whether the feature is common to all OUs
# and whether that feature is required to perform differencing.
DIFF_SCHEMA_METADATA = ["ou_index", "data_id"] + UNIQUE_QUERY_ID_INDEX + [
        # Start Time and End Time of the given OU invocation.
        "start_time", "end_time",

        # (plan_node_id, left_child_plan_node_id, right_child_plan_node_id) are used to reconstruct the plan tree.
        "plan_node_id",
        "left_child_plan_node_id",
        "right_child_plan_node_id",

        "invocation_count",
    ]

# DIFF_SCHEMA_METADATA with all the target columns that need to be differenced.
DIFF_SCHEMA_WITH_TARGETS = DIFF_SCHEMA_METADATA + ["total_cost", "startup_cost"] + TARGET_COLUMNS

# Given a 2D DataFrame following the DIFFERENCING_SCHEMA, these give the column offsets of the plan node,
# the left child plan node, the right child plan node, and where the target columns begin.
DIFF_PLAN_NODE_ID_SCHEMA_INDEX = DIFF_SCHEMA_WITH_TARGETS.index("plan_node_id")
DIFF_LEFT_CHILD_PLAN_NODE_ID_SCHEMA_INDEX = DIFF_SCHEMA_WITH_TARGETS.index("left_child_plan_node_id")
DIFF_RIGHT_CHILD_PLAN_NODE_ID_SCHEMA_INDEX = DIFF_SCHEMA_WITH_TARGETS.index("right_child_plan_node_id")
DIFF_INVOCATION_COUNT_SCHEMA_INDEX = DIFF_SCHEMA_WITH_TARGETS.index("invocation_count")
DIFF_ELAPSED_US_SCHEMA_INDEX = DIFF_SCHEMA_WITH_TARGETS.index("elapsed_us")
DIFF_TARGET_START_SCHEMA_INDEX = DIFF_SCHEMA_WITH_TARGETS.index("total_cost")


class DiffPlanIncompleteSubinvocationException(Exception):
    """
    Exception is raised to indicate that differencing encountered a subinvocation with insufficient
    data. This can happen because certain OU events were dropped.
    """


class DiffPlanInvalidDataException(Exception):
    """
    Exception is raised to indicate that invalid data was encountered. Invalid data can mean that certain
    fields are corrupted or differencing produced invalid/inconsistent results.
    """


class DiffPlanUnsupportedParallelException(Exception):
    """
    Exception is raised to indicate that differencing encountered a subinvocation that indicates parallel
    query execution. As we currently don't support parallel query execution, this exception is raised.
    """
