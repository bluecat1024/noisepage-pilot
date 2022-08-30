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
from behavior.model_workload.utils import compute_frame


logger = logging.getLogger("windowize")


def windowize(input_dir, output_dir, benchmark):
    with open(f"{input_dir}/analysis/keyspaces.pickle", "rb") as f:
        table_attr_map = pickle.load(f)
        attr_table_map = pickle.load(f)
        table_keyspace_map = pickle.load(f)
        query_template_map = pickle.load(f)
        window_index_map = pickle.load(f)

    TABLES = BENCHDB_TO_TABLES[benchmark]
    for tbl in TABLES:
        Path(f"{output_dir}/windows/{tbl}").mkdir(parents=True, exist_ok=True)

    initial_data = {}
    for tbl in TABLES:
        if Path(f"{input_dir}/snippets/chunk_0/{tbl}.feather").exists():
            initial_data[tbl] = pd.read_feather(f"{input_dir}/snippets/chunk_0/{tbl}.feather")
            # This is the data at time t=0.
            initial_data[tbl].to_feather(f"{output_dir}/windows/{tbl}/dataat_0.feather")

    hold_data = {t: [] for t in TABLES}
    hold_index = {t: -1 for t in TABLES}

    key_fn = lambda x: int(x.split("_")[-1])
    containers = sorted(glob.glob(f"{input_dir}/snippets/chunk_*"), key=key_fn)[0:-1]
    for container in tqdm(containers):
        key_num = key_fn(container)
        logger.info("Processing chunk %s", container)
        input_chunk = pd.read_feather(f"{container}/chunk_augment.feather")

        # We've now produced a frame consisting of all the augmentations.
        for cgrp in input_chunk.groupby(by=["target"]):
            # Compute the window index that we care about.
            tbl = cgrp[0]
            df = cgrp[1]
            logger.info("[%s, %s] Processing the combined processed data.", key_num, tbl)

            df.set_index(keys=["unix_timestamp"], inplace=True)
            df.sort_index(axis=0, inplace=True)

            windows = window_index_map[tbl]
            df = pd.merge_asof(df, windows, left_index=True, right_index=True, allow_exact_matches=True)

            # Reset the index and delete all invalid data.
            df.reset_index(drop=False, inplace=True)
            df.drop(df[df.window_index.isna()].index, inplace=True)
            df.reset_index(drop=True, inplace=True)

            for wgrp in df.groupby(by=["window_index"]):
                if hold_index[tbl] == -1:
                    hold_index[tbl] = wgrp[0]
                    hold_data[tbl].append(wgrp[1])
                elif hold_index[tbl] == wgrp[0]:
                    hold_data[tbl].append(wgrp[1])
                else:
                    assert wgrp[0] > 0
                    df = pd.concat(hold_data[tbl], ignore_index=True)

                    if len(df) > 0 and df.iloc[0].true_window_index != -1:
                        # Only write out the dataframe if the true window index is valid.
                        true_window_index = df.iloc[0].true_window_index
                        logger.info("[%s, %s, %s] Writing out window data as true window %s.", key_num, wgrp[0] - 1, tbl, true_window_index)
                        df.to_feather(f"{output_dir}/windows/{tbl}/queries_{true_window_index}.feather")
                    elif len(df) > 0:
                        logger.info("[%s, %s, %s] Found illegal window index.", key_num, wgrp[0] - 1, tbl)

                    # Update the data frame based on queries in this window.
                    pk_keys = []
                    all_keys = []
                    if tbl in table_keyspace_map and tbl in table_keyspace_map[tbl] and tbl in table_attr_map:
                        pk_keys = table_keyspace_map[tbl][tbl]
                        all_keys = table_attr_map[tbl]

                        logger.info("[%s, %s, %s] Computing change frame.", key_num, wgrp[0] - 1, tbl)
                        initial_data[tbl], changed = compute_frame(initial_data[tbl], df, pk_keys, all_keys)
                        if changed and len(df) > 0 and df.iloc[0].true_window_index != -1:
                            # This dataframe represents data after this window has executed.
                            true_window_index = df.iloc[0].true_window_index
                            logger.info("[%s, %s, %s] Writing out new frame state for the next true window %s.", key_num, wgrp[0] - 1, tbl, true_window_index)
                            initial_data[tbl].to_feather(f"{output_dir}/windows/{tbl}/dataat_{true_window_index}.feather")
                    else:
                        logger.info("[%s, %s, %s] Skipping compute change frame.", key_num, wgrp[0] - 1, tbl)

                    hold_index[tbl] = wgrp[0]
                    hold_data[tbl] = [wgrp[1]]

    # Write out the final chunk.
    for tbl, hold in hold_index.items():
        if len(hold_data[tbl]) > 0:
            df = pd.concat(hold_data[tbl], ignore_index=True)
            if len(df) > 0 and df.iloc[0].true_window_index != -1:
                true_window_index = df.iloc[0].true_window_index
                logger.info("[%s, %s, %s] Writing out final window data as true window %s.", key_num, wgrp[0], tbl, true_window_index)
                df.to_feather(f"{output_dir}/windows/{tbl}/queries_{true_window_index}.feather")


class WindowizeCLI(cli.Application):
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

    def main(self):
        b_parts = self.benchmark.split(",")
        input_parts = self.dir_workload_input.split(",")
        output_parts = self.dir_workload_output.split(",")
        for i in range(len(output_parts)):
            logger.info("Processing %s -> %s (%s)", input_parts[i], output_parts[i], b_parts[i])
            windowize(input_parts[i], output_parts[i], b_parts[i])

if __name__ == "__main__":
    Windowize.run()
