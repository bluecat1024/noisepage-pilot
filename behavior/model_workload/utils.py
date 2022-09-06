import numpy as np
import pandas as pd
import pickle


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


class SliceLoader():
    def _load_next_chunk(self):
        self.current_chunk = pd.read_feather(self.files[0])
        self.files = self.files[1:]

    def __init__(self, logger, files, slice_window):
        super(SliceLoader, self).__init__()
        self.files = files
        self.slice_window = slice_window
        self._load_next_chunk()
        self.slice_num = 0
        self.logger = logger

    def _get_from_chunk(self, num):
        assert num <= self.current_chunk.shape[0]
        chunk = self.current_chunk.iloc[:num].copy()
        chunk.reset_index(drop=True, inplace=True)
        self.current_chunk = self.current_chunk.iloc[num:]
        return chunk

    def _num_slice_elements(self, chunk):
        # Computes the number of elements satisfying the "slice" property.
        # This is typically the # of unique statement_timestamp but can also be a tuple.
        return chunk[["statement_timestamp", "pid"]].drop_duplicates().shape[0]

    def _get_num_for_slice_elements(self, chunk, slice_elems):
        if slice_elems >= self._num_slice_elements(chunk):
            return chunk.shape[0]

        c = chunk[["statement_timestamp", "pid"]].drop_duplicates().sort_values(by=["statement_timestamp", "pid"], ignore_index=True)
        ub = c.iloc[slice_elems]

        leq_st = chunk.statement_timestamp < ub.statement_timestamp
        match_st = (chunk.statement_timestamp == ub.statement_timestamp) & (chunk.statement_timestamp < ub.pid)
        return (chunk[leq_st | match_st]).shape[0]

    def get_next_slice(self):
        if self._num_slice_elements(self.current_chunk) >= self.slice_window:
            slice_num = self.slice_num
            self.slice_num = self.slice_num + 1
            return slice_num, self._get_from_chunk(self._get_num_for_slice_elements(self.current_chunk, self.slice_window))

        if self.current_chunk.shape[0] == 0:
            chunk = None
        else:
            # Get the whole chunk since we know we need all of it.
            chunk = self._get_from_chunk(self.current_chunk.shape[0])

        while (chunk is None or self._num_slice_elements(chunk) < self.slice_window) and len(self.files) > 0:
            # Load the next chunk.
            self._load_next_chunk()

            cur_size = 0 if chunk is None else chunk.statement_timestamp.nunique()
            data_slice = self._get_num_for_slice_elements(self.current_chunk, self.slice_window - cur_size)
            next_chunk = self._get_from_chunk(data_slice)
            chunk = pd.concat([chunk, next_chunk], ignore_index=True)

        slice_num = self.slice_num
        self.slice_num = self.slice_num + 1
        return slice_num, chunk
