from plumbum import cli
import gc
import pickle
import glob
from copy import deepcopy
from tqdm import tqdm
import psycopg
import pandas as pd
from pathlib import Path
import shutil
import numpy as np
import json
import logging

from behavior import BENCHDB_TO_TABLES
from behavior.model_workload.utils import compute_frame, keyspace_metadata_read


logger = logging.getLogger("windowize")


def output_window_index(input_dir, initial_data, key_num, window_index, tbl, table_keyspace_map, table_attr_map, df, compute_change):
    if df.shape[0] > 0 and df.iloc[0].true_window_index != -1:
        true_window_index = df.iloc[0].true_window_index
        # Only write out the dataframe if the true window index is valid.
        logger.debug("[%s, %s, %s] Writing out the window data as true window %s.", key_num, window_index, tbl, true_window_index)
        # Get the correct chunk to read and play forward from.
        df.to_feather(f"{input_dir}/windows/{tbl}/queries_{true_window_index}.feather")

        if compute_change and tbl in table_keyspace_map and tbl in table_keyspace_map[tbl] and tbl in table_attr_map:
            # FIXME(UNIQUE): Assume that if we're tracking the changes to a table, that the table has a PK
            # keyspace that we can use for determining whether INSERTs and DELETEs interact.
            pk_keys = table_keyspace_map[tbl][tbl]
            all_keys = table_attr_map[tbl]

            # We want to compute only against the "base" relation data (which has target_index_name.isna()).
            initial_data[tbl], _, changed = compute_frame(initial_data[tbl], df[df.target_index_name.isna()], pk_keys, all_keys, logger)
            if changed:
                logger.debug("[%s, %s, %s] Writing out new frame state for the next true window %s.", key_num, window_index, tbl, true_window_index)
                initial_data[tbl].to_feather(f"{input_dir}/windows/{tbl}/data_{true_window_index}.feather")
            else:
                logger.debug("[%s, %s, %s] No changes found to frame.", key_num, window_index, tbl)
        else:
            logger.debug("[%s, %s, %s] Skipping compute change frame.", key_num, window_index, tbl)
    elif df.shape[0] > 0:
        logger.debug("[%s, %s, %s] Found illegal window index.", key_num, window_index, tbl)


def windowize(input_dir):
    table_attr_map, _, table_keyspace_map, _, _, window_index_map = keyspace_metadata_read(f"{input_dir}/analysis")

    tables = table_attr_map.keys()
    initial_data = {}
    for tbl in tables:
        Path(f"{input_dir}/windows/{tbl}").mkdir(parents=True, exist_ok=True)
        if Path(f"{input_dir}/snippets/chunk_0/{tbl}.feather").exists():
            initial_data[tbl] = pd.read_feather(f"{input_dir}/snippets/chunk_0/{tbl}.feather")
            initial_data[tbl].to_feather(f"{input_dir}/windows/{tbl}/data_0.feather")

    hold_data = {t: [] for t in tables}
    hold_index = {t: -1 for t in tables}

    key_fn = lambda x: int(x.split("_")[-1])
    containers = sorted(glob.glob(f"{input_dir}/snippets/chunk_*"), key=key_fn)
    for container in tqdm(containers):
        key_num = key_fn(container)
        logger.debug("Processing chunk %s", container)
        input_chunk = pd.read_feather(f"{container}/augment_chunk.feather")

        # We've now produced a frame consisting of all the augmentations.
        for cgrp in input_chunk.groupby(by=["target"]):
            # Compute the window index that we care about.
            tbl = cgrp[0]
            df = cgrp[1]
            logger.debug("[%s, %s] Processing the combined processed data.", key_num, tbl)

            df.set_index(keys=["unix_timestamp"], inplace=True)
            df.sort_index(axis=0, inplace=True)

            # Merge to find the correct window.
            windows = window_index_map[tbl]
            df = pd.merge_asof(df, windows, left_index=True, right_index=True, allow_exact_matches=True)

            # Reset the index and delete all invalid data.
            df.reset_index(drop=False, inplace=True)
            df.drop(df[df.window_index.isna()].index, inplace=True)
            df.reset_index(drop=True, inplace=True)

            for wgrp in df.groupby(by=["window_index"]):
                if hold_index[tbl] == -1:
                    # No window index has been constructed yet.
                    hold_index[tbl] = wgrp[0]
                    hold_data[tbl].append(wgrp[1])
                elif hold_index[tbl] == wgrp[0]:
                    # Same window index data.
                    hold_data[tbl].append(wgrp[1])
                else:
                    # We are now at a new window index.
                    assert wgrp[0] > 0
                    df = pd.concat(hold_data[tbl], ignore_index=True)
                    df.sort_values(by=["query_order"], inplace=True, ignore_index=True)
                    output_window_index(input_dir, initial_data, key_num, hold_index[tbl], tbl, table_keyspace_map, table_attr_map, df, True)
                    del df

                    # Install the new window index.
                    hold_index[tbl] = wgrp[0]
                    hold_data[tbl] = [wgrp[1]]

    # Write out the final chunk.
    for tbl, hold in hold_index.items():
        if len(hold_data[tbl]) > 0:
            df = pd.concat(hold_data[tbl], ignore_index=True)
            output_window_index(input_dir, initial_data, key_num, hold_index[tbl], tbl, table_keyspace_map, table_attr_map, df, False)


class WindowizeCLI(cli.Application):
    dir_workload_input = cli.SwitchAttr(
        "--dir-workload-input",
        str,
        mandatory=True,
        help="Path to the folder containing the workload input.",
    )

    def main(self):
        input_parts = self.dir_workload_input.split(",")
        for i in range(len(input_parts)):
            logger.info("Processing %s", input_parts[i])
            windowize(input_parts[i])

if __name__ == "__main__":
    Windowize.run()
