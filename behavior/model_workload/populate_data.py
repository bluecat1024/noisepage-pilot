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
from behavior.model_workload.utils import compute_frames, compute_frame, keyspace_metadata_read, SliceLoader

logger = logging.getLogger("populate_data")


def load_initial_data(input_dir, workload_only, psycopg2_conn, table_attr_map):
    data_map = {}

    # Populate the initial data map that we care about.
    # This reads either from postgres or from the CSV files, extracting columns we want.
    if workload_only:
        with psycopg.connect(psycopg2_conn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                for tbl in table_attr_map.keys():
                    if len(table_attr_map[tbl]) == 0:
                        logger.info("Skipping read from [%s]", tbl)
                        continue

                    if Path(f"{input_dir}/snippets/chunk_0/{tbl}.feather").exists():
                        logger.info("Reading [%s] from file", tbl)
                        frame = pd.read_feather(f"{input_dir}/snippets/chunk_0/{tbl}.feather")
                        data_map[tbl] = frame
                        continue

                    pks_sel = ",".join(table_attr_map[tbl])
                    logger.info("Reading %s from [%s]", pks_sel, tbl)

                    result = cursor.execute(f"SELECT {pks_sel} FROM {tbl}", prepare=False)
                    frame = pd.DataFrame(result, columns=table_attr_map[tbl])
                    frame.to_feather(f"{input_dir}/snippets/chunk_0/{tbl}.feather")
                    data_map[tbl] = frame
    else:
        for tbl in table_attr_map.keys():
            if len(table_attr_map[tbl]) == 0:
                logger.info("Skipping read from [%s]", tbl)
                continue

            logger.info("Reading from [%s]", tbl)
            if Path(f"{input_dir}/snippets/chunk_0/{tbl}.feather").exists():
                frame = pd.read_feather(f"{input_dir}/snippets/chunk_0/{tbl}.feather")
                data_map[tbl] = frame
                continue

            assert Path(f"{input_dir}/{tbl}_snapshot.csv").exists()
            input_frame = pd.read_csv(f"{input_dir}/{tbl}_snapshot.csv")
            input_frame = input_frame[table_attr_map[tbl]]
            input_frame.to_feather(f"{input_dir}/snippets/chunk_0/{tbl}.feather")
            data_map[tbl] = input_frame

    return data_map


def build_augment_chunk(container, input_chunk, marked_slots, workload_only):
    # Change the index to query_order which is guaranteed to be unique.
    input_chunk.set_index(keys=["query_order"], inplace=True)
    marked_slots.set_index(keys=["query_order"], inplace=True)

    chunk_contain_slots = input_chunk[input_chunk.index.isin(marked_slots.index.unique())]
    resolved_slots = input_chunk[~input_chunk.index.isin(chunk_contain_slots.index)]
    resolved_slots.reset_index(drop=False, inplace=True)
    if workload_only:
        targets = resolved_slots.target.str.split(',').apply(pd.Series, 1).stack()
        targets = targets.replace('', np.nan).dropna()
        targets.index = targets.index.droplevel(-1)
        targets.name = "target"

        pre_num_slots = resolved_slots.shape[0]
        resolved_slots.drop(columns=["target"], inplace=True)
        resolved_slots = resolved_slots.join(targets, how="inner")

        # This simple assumption is such that for now, we ensure that we are correcting ",[tbl]" -> "[tbl]".
        # This assert *does* not imply something is wrong, it's just that only the above case is checked.
        assert pre_num_slots == resolved_slots.shape[0]

    # Remove all columns from the chunk of slots that need augmentation.
    chunk_contain_slots = chunk_contain_slots.drop(columns=[c for c in marked_slots.columns], errors='ignore')
    chunk_contain_slots = chunk_contain_slots.join(marked_slots, how="inner")
    chunk_contain_slots.reset_index(drop=False, inplace=True)

    if "target_table" in chunk_contain_slots:
        # Reassign target_table into target.
        chunk_contain_slots["target"] = chunk_contain_slots["target_table"]
        chunk_contain_slots.drop(columns=["target_table"], inplace=True)

    # Create the final chunk.
    input_chunk = pd.concat([resolved_slots, chunk_contain_slots], ignore_index=True)

    # Write out the augmented chunk.
    input_chunk.to_feather(f"{container}/augment_chunk.feather")

    del input_chunk
    gc.collect()


def discover_multi_matches(tbl, invalids, data_map, table_attr_map, attr_table_map, query_template_map):
    s = data_map[tbl].copy()

    equi_keys = []
    high_keys = []
    loweq_keys = []
    for key in table_attr_map[tbl]:
        if key in invalids:
            dtype = invalids.dtypes[key]
            found = False
            if (dtype == "object" and np.sum(invalids[key].isnull()) == 0) or (dtype == "float64" and np.sum(invalids[key].isna()) == 0):
                # This should be used as an equi-join key.
                equi_keys.append(key)
                found = True

            high = key + "_high"
            loweq = key + "_loweq"
            if high in invalids and np.sum(invalids[high].isna()) == 0:
                high_keys.append(key)
                found = True

            if loweq in invalids and np.sum(invalids[loweq].isna()) == 0:
                loweq_keys.append(key)
                found = True

            if not found and (invalids.iloc[0].query_text) in query_template_map and key in query_template_map[invalids.iloc[0].query_text]:
                # This is to identify cases where the column is supplied by another table.
                value = query_template_map[invalids.iloc[0].query_text][key]
                if value in attr_table_map and attr_table_map[value] != tbl:
                    other_tbl = attr_table_map[value]
                    # Not allowed to recurse again.
                    other_s = discover_multi_matches(other_tbl, invalids, data_map, table_attr_map, attr_table_map, {})

                    # Match based on the key and the value. We want the query_order transplant.
                    # This is to ensure that query_order from a previous table is carried forwards to the next table.
                    other_s.drop(columns=[c for c in other_s if c != "query_order" and c != value], inplace=True)
                    s = s.merge(other_s, left_on=[key], right_on=[value])
                    s.reset_index(drop=True, inplace=True)
                    equi_keys = ["query_order"] + equi_keys

    total_keys = equi_keys + [k + "_high" for k in high_keys] + [k + "_loweq" for k in loweq_keys] + ["query_order"]
    invalids = invalids.drop(columns=[c for c in invalids if c not in total_keys])
    invalids.set_index(keys=equi_keys, inplace=True)
    s = s.merge(invalids, on=equi_keys, how="inner")
    assert not any([c.endswith("_x") for c in s]) and not any([c.endswith("_y") for c in s])

    for key in high_keys:
        s = s[s[key] < s[key + "_high"]]

    for key in loweq_keys:
        s = s[s[key] >= s[key + "_loweq"]]

    return s


def populate_data(input_dir, slice_window, workload_only, psycopg2_conn, skip_save_frames):
    Path(f"{input_dir}/snippets/chunk_0").mkdir(parents=True, exist_ok=True)
    table_attr_map, attr_table_map, table_keyspace_map, query_template_map, indexoid_name_map, _ = keyspace_metadata_read(f"{input_dir}/analysis")

    # Load the initial data map.
    data_map = load_initial_data(input_dir, workload_only, psycopg2_conn, table_attr_map)
    key_fn = lambda x: int(x.split(".feather")[0].split("_")[-1])
    files = sorted(glob.glob(f"{input_dir}/analysis/*.feather"), key=key_fn)
    loader = SliceLoader(logger, files, slice_window)

    logger.info("Starting to populate the data.")
    chunk_num, chunk = loader.get_next_slice()
    with tqdm() as pbar:
        while chunk is not None:
            # Create the directory.
            Path(f"{input_dir}/snippets/chunk_{chunk_num}").mkdir(parents=True, exist_ok=True)

            touched_tbls = {t: False for t in table_attr_map.keys()}
            augmented_slots = []
            augmented_tbl_pk_outputs = {t: [] for t in table_attr_map.keys()}

            join_map = {}

            # Base relation is where target_index_name is None. We only consider the base relation data
            # when considering the modifications to present table data.
            compute = chunk[chunk.target_index_name.isna()]
            compute_frames(compute, data_map, join_map, touched_tbls, table_attr_map, table_keyspace_map, logger)

            for group in chunk.groupby(by=["query_id", "target"]):
                tbl_slices = group[0][1]
                for tbl in tbl_slices.split(","):
                    if len(tbl) == 0:
                        # This is a side-effect of concatenation failing.
                        continue

                    if tbl not in table_attr_map or tbl not in table_keyspace_map or tbl not in table_keyspace_map[tbl]:
                        # We require that a PK exists. If a PK doesn't exist, then there is nothing worth looking at.
                        # See above: we use the PK for determinig how data within a table evolves over time.
                        continue

                    invalid_slots = None
                    data_chunk = group[1]
                    need_aug = False
                    for (keyspace_name, keyspace) in table_keyspace_map[tbl].items():
                        # In principle, we don't actually care about whether the keyspace is against the base or not.
                        # This is because we want the fully specified keyspace domain in any EVENT that the query
                        # may affect a particular keyspace anyways...
                        if workload_only:
                            invalids = data_chunk
                            need_aug |= (np.sum(invalids[keyspace].isna().any(axis=1)) > 0)
                        else:
                            invalids = data_chunk[data_chunk[keyspace].isna().any(axis=1)]
                            need_aug = True

                        # If there are any invalids, then join with the other keyspace invalids.
                        # Since we are grouped by query already, all the valid parameters are the same.
                        # Finding a single column for a missing keyspace is the same as finding for all.
                        if invalids.shape[0] > 0:
                            if invalid_slots is None:
                                invalid_slots = invalids.index
                            else:
                                invalid_slots = invalid_slots.union(invalids.index)

                    if invalid_slots is not None and invalid_slots.shape[0] > 0:
                        invalids = data_chunk.loc[invalid_slots]

                        # We have missing data that we need to populate. Use the join_map since we want to capture the
                        # "worst-case" possible number of matching data entries.
                        s = discover_multi_matches(tbl, invalids, join_map, table_attr_map, attr_table_map, query_template_map)

                        if need_aug:
                            if workload_only:
                                # Attach a target_table to disambiguate SELECTs with multiple tables.
                                s["target_table"] = tbl

                            # Only attach to augmented slots if we need the augmented data.
                            augmented_slots.append(s)

                        if workload_only:
                            # Use the value_counts() on s.query_order in order to find "affected tuples".
                            vc = s.query_order.value_counts()
                            vc.name = f"{tbl}_pk_output"
                            augmented_tbl_pk_outputs[tbl].append(vc)

            if workload_only:
                chunk["num_modify"] = 0
                for tbl, vecs in augmented_tbl_pk_outputs.items():
                    if len(vecs) > 0:
                        # Join and sort the index in ascending order.
                        # This is so the chunk.loc[] order is correct.
                        df = pd.concat(vecs)
                        df.sort_index(inplace=True)

                        match = chunk.query_order.isin(df.index)
                        chunk.loc[match, df.name] = df.values
                        chunk.loc[match, "num_modify"] = chunk.loc[match, "num_modify"] + df.values

                if not skip_save_frames:
                    # Only write out frames if requested.
                    for tbl, touched in touched_tbls.items():
                        if touched:
                            Path(f"{input_dir}/snippets/chunk_{chunk_num+1}").mkdir(parents=True, exist_ok=True)
                            data_map[tbl].to_feather(f"{input_dir}/snippets/chunk_{chunk_num+1}/{tbl}.feather")
            chunk.to_feather(f"{input_dir}/snippets/chunk_{chunk_num}/chunk.feather")

            # Process the augmentations.
            if len(augmented_slots) > 0:
                df = pd.concat(augmented_slots, ignore_index=True)
                build_augment_chunk(f"{input_dir}/snippets/chunk_{chunk_num}/", chunk, df, workload_only)
                del df

            gc.collect()

            chunk_num, chunk = loader.get_next_slice()
            pbar.update()

    if workload_only and Path(f"{input_dir}/snippets/chunk_{chunk_num}").exists():
        shutil.rmtree(f"{input_dir}/snippets/chunk_{chunk_num}")


class PopulateDataCLI(cli.Application):
    dir_workload_input = cli.SwitchAttr(
        "--dir-workload-input",
        str,
        mandatory=True,
        help="Path to the folder containing the workload input.",
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

    slice_window = cli.SwitchAttr(
        "--slice-window",
        int,
        help="Size of the window slice to use for analysis.",
    )

    skip_save_frames = cli.Flag("--skip-save-frames", default=False, help="Whether to skip saving the intermediate data frames.")

    def main(self):
        input_parts = self.dir_workload_input.split(",")
        for i in range(len(input_parts)):
            logger.info("Processing %s (%s)", input_parts[i], self.workload_only)
            populate_data(input_parts[i], self.slice_window, (self.workload_only == "True"), self.psycopg2_conn, self.skip_save_frames)


if __name__ == "__main__":
    PopulateDataCLI.run()
