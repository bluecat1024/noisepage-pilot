import numpy as np
import pandas as pd
import pickle
from enum import Enum


class OpType(Enum):
    SELECT = 1
    INSERT = 2
    UPDATE = 3
    DELETE = 4


def keyspace_metadata_output(container, *args):
    with open(f"{container}/keyspaces.pickle", "wb") as f:
        for arg in args:
            pickle.dump(arg, f)


def keyspace_metadata_read(container):
    args = []
    with open(f"{container}/keyspaces.pickle", "rb") as f:
        while True:
            try:
                args.append(pickle.load(f))
            except:
                break

    return tuple(args)


def compute_frame(frame, deltas, pk_keys, all_keys, logger=None):
    # FIXME(UPDATE): Assume UPDATEs do not effect the primary keys of the tuples that they change.
    # We actually make a larger statement: UPDATEs don't change ANY keys whatsoever and so
    # don't require index inserts if a HOT is determined to be valid. Else we need both the
    # "old" keys as DELETE and the "new" keys as INSERT.
    insert_index = deltas[deltas.is_insert].index
    delete_index = deltas[deltas.is_delete].index
    if len(insert_index) == 0 and len(delete_index) == 0:
        return frame, frame, False

    ins_deltas = deltas.loc[insert_index, all_keys + ["query_order"]]
    del_deltas = deltas.loc[delete_index, all_keys + ["query_order"]]
    del_deltas["del_index"] = del_deltas.index
    ins_deltas.set_index(keys=["query_order"], inplace=True)
    del_deltas.set_index(keys=["query_order"], inplace=True)

    # For each INSERT delta, see if there is a matching PK KEY delete that happened afterwards.
    ins = pd.merge_asof(ins_deltas, del_deltas, left_index=True, right_index=True, by=pk_keys, suffixes=(None, "_del"), direction='forward')
    # An invalid del_index means that the inserted tuple survives.
    ins.drop(columns=[c for c in ins if c.endswith("_del")], inplace=True)
    ins = ins[ins.del_index.isna()]

    if ins_deltas.shape[0] > 0:
        # Apply all the INSERTs. This should be the JOIN frame.
        join_frame = pd.concat([frame, ins_deltas], ignore_index=True)
    else:
        # Otherwise copy the frame out for the JOIN frame since we might modify frame later.
        join_frame = frame.copy()

    if del_deltas.shape[0] > 0:
        # Now apply all the DELETES.
        del_deltas.set_index(keys=pk_keys, inplace=True)
        frame.set_index(keys=pk_keys, inplace=True)
        frame = frame[~frame.index.isin(del_deltas.index)]
        frame.reset_index(drop=False, inplace=True)

    if ins.shape[0] > 0:
        if logger is not None and ins.shape[0] != ins_deltas.shape[0]:
            diff = ins_deltas.index.difference(ins.index)
            logger.warning("Noticed that we have some inserts being deleted in the same window. %s", ins_deltas.loc[diff])
        ins.drop(columns=["del_index"], inplace=True)
        frame = pd.concat([frame, ins], ignore_index=True)

    return frame, join_frame, (del_deltas.shape[0] > 0) or (ins_deltas.shape[0] > 0)


def compute_frames(chunk, data_map, join_map, touched_tbls, table_attr_map, table_keyspace_map, logger):
    # Computes an update step based on queries. Assumes that target column exists.
    for group in chunk.groupby(by=["target"]):
        # FIXME(MODIFY): Assume that we all modification queries impact only 1 table.
        tbls = [t for t in group[0].split(",") if len(t) > 0]
        for tbl in tbls:
            if tbl not in table_keyspace_map or tbl not in table_keyspace_map[tbl]:
                # No valid keys that are worth looking at.
                continue

            # FIXME(UNIQUE): Assume that if we're tracking the changes to a table, that the table has a PK
            # keyspace that we can use for determining whether INSERTs and DELETEs interact.
            pk_keys = table_keyspace_map[tbl][tbl]

            # FIXME(INSERT/DELETE): We also assume that for INSERT and DELETE, PK is fully specified.
            insdel = group[1][(group[1].is_insert | group[1].is_delete)]
            if insdel.shape[0] > 0:
                invalids = insdel[insdel[pk_keys].isna().any(axis=1)]
                assert invalids.shape[0] == 0

            all_keys = table_attr_map[tbl]
            data_map[tbl], join_map[tbl], touched = compute_frame(data_map[tbl], group[1], pk_keys, all_keys, logger)
            touched_tbls[tbl] |= touched
