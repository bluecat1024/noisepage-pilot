import gc
import pickle
import glob
from tqdm import tqdm
import psycopg
import pandas as pd
from pathlib import Path
import shutil
import numpy as np
import logging
from plumbum import cli
from behavior import BENCHDB_TO_TABLES
from behavior.model_workload.utils import compute_frame

TABLES = []
logger = logging.getLogger("populate_data")


def load_initial_data(input_dir, output_dir, workload_only, psycopg2_conn, table_attr_map):
    data_map = {}

    # Populate the initial data map that we care about.
    # This reads either from postgres or from the CSV files, extracting columns we want.
    if workload_only:
        with psycopg.connect(psycopg2_conn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                for tbl in TABLES:
                    if len(table_attr_map[tbl]) == 0:
                        logger.info("Skipping read from [%s]", tbl)
                        continue

                    if Path(f"{output_dir}/snippets/chunk_0/{tbl}.feather").exists():
                        logger.info("Reading [%s] from file", tbl)
                        frame = pd.read_feather(f"{output_dir}/snippets/chunk_0/{tbl}.feather")
                        data_map[tbl] = frame
                        continue

                    pks_sel = ",".join(table_attr_map[tbl])
                    logger.info("Reading %s from [%s]", pks_sel, tbl)

                    result = cursor.execute(f"SELECT {pks_sel} FROM {tbl}", prepare=False)
                    frame = pd.DataFrame(result, columns=table_attr_map[tbl])
                    frame.to_feather(f"{output_dir}/snippets/chunk_0/{tbl}.feather")
                    data_map[tbl] = frame
    else:
        for tbl in TABLES:
            if len(table_attr_map[tbl]) == 0:
                logger.info("Skipping read from [%s]", tbl)
                continue

            logger.info("Reading from [%s]", tbl)
            if Path(f"{output_dir}/snippets/chunk_0/{tbl}.feather").exists():
                frame = pd.read_feather(f"{output_dir}/snippets/chunk_0/{tbl}.feather")
                data_map[tbl] = frame
                continue

            assert Path(f"{input_dir}/{tbl}_snapshot.csv").exists()
            input_frame = pd.read_csv(f"{input_dir}/{tbl}_snapshot.csv")
            input_frame = input_frame[table_attr_map[tbl]]
            input_frame.to_feather(f"{output_dir}/snippets/chunk_0/{tbl}.feather")
            data_map[tbl] = input_frame

    return data_map


def construct_augmented_chunk(key_num, container, workload_only):
    input_chunk = pd.read_feather(f"{container}/chunk.feather")

    def merge_chunk(input_chunk, augment_chunk):
        augment_chunk.set_index(keys=["slot"], inplace=True)
        input_chunk.set_index(keys=["slot"], inplace=True)

        # Find all slots that we need to mutate.
        augment_input = input_chunk[input_chunk.index.isin(augment_chunk.index.unique())]
        input_chunk = input_chunk[~input_chunk.index.isin(augment_input.index)]
        input_chunk.reset_index(drop=False, inplace=True)

        # Remove all the bad columns from the input chunk.
        augment_input = augment_input.drop(columns=[c for c in augment_chunk.columns if c != 'target'], errors='ignore')
        augment_input.reset_index(drop=False, inplace=True)
        augment_chunk.reset_index(drop=False, inplace=True)

        # Combine and then attach back to the original chunk.
        join_key = ["slot"] if workload_only else ["slot", "target"]
        augment_input.set_index(keys=join_key, inplace=True)
        augment_chunk.set_index(keys=join_key, inplace=True)
        augment_chunk = augment_input.join(augment_chunk, how="inner", rsuffix="_remove")
        augment_chunk.reset_index(drop=False, inplace=True)
        input_chunk = pd.concat([input_chunk, augment_chunk], ignore_index=True)

        remove = [c for c in input_chunk.columns if c.endswith("_remove")]
        assert len(remove) <= 1
        if len(remove) == 1:
            input_chunk.loc[~input_chunk.target_remove.isna(), "target"] = input_chunk.loc[~input_chunk.target_remove.isna(), "target_remove"]
            input_chunk.drop(columns=remove, inplace=True)

        del augment_chunk
        return input_chunk

    logger.info("[%s] Merging chunk with augmented data.", key_num)
    input_chunk = merge_chunk(input_chunk, pd.read_feather(f"{container}/augment.feather"))
    for tbl in TABLES:
        if Path(f"{container}/deferred_{tbl}.feather").exists():
            logger.info("[%s] Merging chunk with deferred %s data.", key_num, tbl)
            defer = pd.read_feather(f"{container}/deferred_{tbl}.feather")
            input_chunk = merge_chunk(input_chunk, defer)

    # Write out the augmented chunk.
    input_chunk.to_feather(f"{container}/chunk_augment.feather")

    del input_chunk
    gc.collect()


def discover_matches(slot, offending, data_map, table_attr_map, attr_table_map, query_template_map, tbl):
    s = data_map[tbl]
    for key in table_attr_map[tbl]:
        if key in offending:
            high = key + "_high"
            loweq = key + "_loweq"

            found = False
            if isinstance(offending[key], str) and offending[key] is not None:
                # Case of equality on a string predicate.
                s = s[s[key] == offending[key]]
                found = True
            elif offending[key] is not None and not np.isnan(offending[key]):
                # Case of equality on a numeric predicate.
                s = s[s[key] == offending[key]]
                found = True

            if high in offending and not np.isnan(offending[high]):
                # Case of exclusive high key on numeric predicate.
                s = s[s[key] < offending[high]]
                found = True
            if loweq in offending and not np.isnan(offending[loweq]):
                # Case of inclusive low key on numeric predicate.
                s = s[s[key] >= offending[loweq]]
                found = True

            if not found and key in query_template_map[offending.query_text]:
                # This is to identify cases where the column is supplied by another table.
                value = query_template_map[offending.query_text][key]
                if value in attr_table_map and attr_table_map[value] != tbl:
                    other_s = data_map[attr_table_map[value]]
                    # Try to filter the other table using valid keys.
                    for skey in table_attr_map[attr_table_map[value]]:
                        if skey in offending:
                            high = skey + "_high"
                            loweq = skey + "_loweq"

                            if isinstance(offending[skey], str) and offending[skey] is not None:
                                # We have an equality string key on the other column.
                                other_s = other_s[other_s[skey] == offending[skey]]
                            elif not np.isnan(offending[skey]):
                                # We have an equality numeric key on the other column.
                                other_s = other_s[other_s[skey] == offending[skey]]

                            if high in offending and not np.isnan(offending[high]):
                                # We have an exclusive high key on the other column.
                                other_s = other_s[other_s[skey] < offending[high]]
                            if loweq in offending and not np.isnan(offending[loweq]):
                                # We have an inclusive low key on the other column.
                                other_s = other_s[other_s[skey] >= offending[loweq]]
                    # Perform the "join" operation.
                    s = s[s[key].isin(other_s[value].values)]

    if s.shape[0] > 0:
        # Indicate that we've found some data if there exists any data at all.
        s["slot"] = slot
        s["target"] = tbl
    return s


def populate_data(benchmark, input_dir, output_dir, workload_only, psycopg2_conn):
    global TABLES
    assert benchmark in BENCHDB_TO_TABLES
    TABLES = BENCHDB_TO_TABLES[benchmark]
    Path(f"{output_dir}/snippets/chunk_0").mkdir(parents=True, exist_ok=True)

    with open(f"{input_dir}/analysis/keyspaces.pickle", "rb") as f:
        table_attr_map = pickle.load(f)
        attr_table_map = pickle.load(f)
        table_keyspace_map = pickle.load(f)
        query_template_map = pickle.load(f)
        window_index_map = pickle.load(f)

    # Load the initial data map.
    data_map = load_initial_data(input_dir, output_dir, workload_only, psycopg2_conn, table_attr_map)

    key_fn = lambda x: int(x.split(".feather")[0].split("_")[-1])
    files = sorted(glob.glob(f"{input_dir}/analysis/*.feather"), key=key_fn)
    for input_file in files:
        file_key = key_fn(input_file)
        Path(f"{output_dir}/snippets/chunk_{file_key}").mkdir(parents=True, exist_ok=True)
        logger.info("Reading input file: %s", input_file)
        input_chunk = pd.read_feather(input_file)
        input_chunk["is_insert"] = (input_chunk.OP == "INSERT").astype(bool)
        input_chunk["is_delete"] = (input_chunk.OP == "DELETE").astype(bool)
        input_chunk["slot"] = input_chunk.index

        slices = {}
        if workload_only:
            for tbl in TABLES:
                # Slice the input query stream based on whether or not the table is used.
                # We want JOINs to map to multiple slices.
                slices[tbl] = input_chunk[input_chunk.query_text.str.contains(tbl)]
        else:
            for tbl in TABLES:
                # Target should be a single value that indicates what table is being accessed.
                slices[tbl] = input_chunk[input_chunk.target == tbl]
        logger.info("[%s, Slices] Finished building slices based on table", file_key)

        # Deferred slots are special case where there are no deltas on the table.
        logger.info("[%d] Started building invalid slots", file_key)
        deferred_slots = {}
        invalid_slots = None
        for tbl in TABLES:
            if tbl in table_keyspace_map and len(table_keyspace_map[tbl]) > 0 and len(table_keyspace_map[tbl][tbl]) > 0:
                # FIXME(INDEX): We only consider the base relation. This is because the base relation determines the
                # HOT percentages which have a much more significant impact on overall query error. For simplicity
                # and proof of concept, we only use the base relation.
                base_pks = table_keyspace_map[tbl][tbl]
                index = slices[tbl][slices[tbl][base_pks].isna().any(axis=1)].index
                if len(index) > 0:
                    single = np.sum(input_chunk.iloc[index].num_rel_refs == 1) == len(index)
                    slots = (input_chunk.is_insert | input_chunk.is_delete) & (input_chunk.modify_target == tbl) & (input_chunk.slot > index[0])
                    if single and np.sum(slots) == 0:
                        # There are no relevant changes so add to the deferred slots.
                        deferred_slots[tbl] = (tbl, index)
                        logger.info("[%d, Slots] Found %s to have %d deferred slots", file_key, tbl, len(index))
                    else:
                        logger.info("[%d, Slots] Found %s to have %d invalid slots", file_key, tbl, len(index))
                        # Add to the "combined" invalid slots.
                        if invalid_slots is None:
                            invalid_slots = index
                        else:
                            invalid_slots = invalid_slots.union(index)
        logger.info("[%d] Finished building invalid slots", file_key)

        # Mutate the data map based on the modifications in the range.
        def process_range(prev_range):
            for group in prev_range.groupby(by=["modify_target"]):
                tbl = group[0]
                if tbl not in table_attr_map or tbl not in table_keyspace_map or tbl not in table_keyspace_map[tbl]:
                    # No valid keys that are worth looking at.
                    continue

                pk_keys = table_keyspace_map[tbl][tbl]
                all_keys = table_attr_map[tbl]
                data_map[tbl], _ = compute_frame(data_map[tbl], group[1], pk_keys, all_keys)

        augmented_slots = []

        if workload_only:
            input_chunk["num_modify"] = 0.0
            logger.info("[%s, Process] Starting to process %s slots", file_key, input_chunk.shape[0])
            it = tqdm(range(input_chunk.shape[0]))
        else:
            logger.info("[%s, Process] Starting to process %s invalid slots", file_key, len(invalid_slots))
            it = enumerate(tqdm(invalid_slots, total=len(invalid_slots)))

        for slot in it:
            # Previous range.
            if workload_only:
                offending = input_chunk.iloc[slot]
                if offending.is_insert:
                    process_range(input_chunk.iloc[slot:slot+1])
                    input_chunk.at[slot, "num_modify"] = 1.0
                    continue

                if offending.is_delete:
                    # We need to get the number of matching deletes to populate num_modify.
                    target = offending.modify_target
                    s = discover_matches(slot, offending, data_map, table_attr_map, attr_table_map, query_template_map, target)

                    process_range(input_chunk.iloc[slot:slot+1])
                    input_chunk.at[slot, "num_modify"] = s.shape[0]
                    continue
            else:
                i, slot = slot[0], slot[1]
                prev_slot = 0 if i == 0 else invalid_slots[i-1] + 1
                prev_range = input_chunk.iloc[prev_slot:slot]
                process_range(prev_range)

                # FIXME(UNDERSPECIFIED): Assume that only SELECTs/UPDATES can be underspecified.
                # This fundamentally assumes that UPDATEs are not key-changing.
                offending = input_chunk.iloc[slot]
                assert (offending.OP == "SELECT") or (offending.OP == "UPDATE")

            tbls = []
            if workload_only:
                # Figure out which table this slot is from. If this becomes super slow, we might need to change
                # the iteration order somehow.
                for tbl in TABLES:
                    if slot in slices[tbl].index:
                        tbls.append(tbl)
                assert len(tbls) > 0
            else:
                tbls = [offending.target]

            # We are now looking at this particular table in the target.
            for tbl in tbls:
                s = discover_matches(slot, offending, data_map, table_attr_map, attr_table_map, query_template_map, tbl)
                if workload_only:
                    # We conveniently choose to set the {tbl}_pk_output based on the number of matches found.
                    input_chunk.at[slot, f"{tbl}_pk_output"] = s.shape[0]

                # Only add if the slot is invalid and not in deferred.
                if s.shape[0] > 0 and slot in invalid_slots and (tbl not in deferred_slots or slot not in deferred_slots[tbl][1]):
                    # Add to the augmented slots.
                    augmented_slots.append(s)

        # Process the remaining component of the range.
        if not workload_only:
            logger.info("[%s, Process] Processing final range", file_key)
            prev_slot = 0 if len(invalid_slots) == 0 else (invalid_slots[-1] + 1)
            prev_range = input_chunk.iloc[prev_slot:]
            process_range(prev_range)

        # Process the deferred special slots.
        for (tbl, slots) in deferred_slots.values():
            logger.info("[%s, Process] Processing deferred slots (%s, %s)", file_key, tbl, len(slots))
            for sgrp in input_chunk.iloc[slots].groupby(by=["query_text"]):
                valid_keys = []
                for key in table_attr_map[tbl]:
                    if key in sgrp[1]:
                        num_na = np.sum(sgrp[1][key].isna())
                        assert num_na == 0 or num_na == sgrp[1].shape[0]
                        if num_na == 0:
                            valid_keys.append(key)

                s = data_map[tbl].set_index(keys=valid_keys, inplace=False)
                s.sort_index(axis=0, inplace=True)
                s.drop(columns=["slot", "target"], inplace=True, errors='ignore')
                orig = sgrp[1][valid_keys + ["slot"]].set_index(valid_keys)
                c = s.join(orig, how="inner")
                c["target"] = tbl
                c.reset_index(drop=False).to_feather(f"{output_dir}/snippets/chunk_{file_key}/deferred_{tbl}.feather")

                if workload_only:
                    match_found = orig.index.isin(s.index)
                    valid = orig[match_found]
                    invalid = orig[~match_found]

                    # Populate the {tbl}_pk_output accordingly.
                    if len(invalid) > 0:
                        input_chunk.loc[invalid.index, f"{tbl}_pk_output"] = 0

                    if len(valid) > 0:
                        for grp in valid.join(s, how="inner").groupby(by=["slot"]):
                            input_chunk.loc[grp[0], f"{tbl}_pk_output"] = grp[1].shape[0]

        if workload_only:
            # Concatenate all the {tbl}_pk_output into a num_modify.
            for tbl in TABLES:
                input_chunk["num_modify"] = input_chunk.num_modify + input_chunk[f"{tbl}_pk_output"].fillna(0.0)

        logger.info("[%s, Output] Outputting all relevant data", file_key)
        input_chunk.to_feather(f"{output_dir}/snippets/chunk_{file_key}/chunk.feather")
        if len(augmented_slots) > 0:
            df = pd.concat(augmented_slots, ignore_index=True)
            df.to_feather(f"{output_dir}/snippets/chunk_{file_key}/augment.feather")
            del df
        del input_chunk
        del augmented_slots
        gc.collect()

        # Construct the augmented chunk from the pieces.
        construct_augmented_chunk(file_key, f"{output_dir}/snippets/chunk_{file_key}/", workload_only)


class PopulateDataCLI(cli.Application):
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
            populate_data(b_parts[i], input_parts[i], output_parts[i], (self.workload_only == "True"), self.psycopg2_conn)


if __name__ == "__main__":
    PopulateDataCLI.run()
