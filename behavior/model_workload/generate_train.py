from scipy import stats
import re
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
from plumbum import cli

from behavior import BENCHDB_TO_TABLES
from behavior.model_workload.model import WorkloadModel
from behavior.model_workload.utils import keyspace_metadata_read

logger = logging.getLogger("gen_train")


def gen_train(input_dir):
    table_attr_map, attr_table_map, table_keyspace_map, _, _, window_index_map = keyspace_metadata_read(f"{input_dir}/analysis")
    tables = table_attr_map.keys()

    for tbl in tables:
        window_index = window_index_map[tbl]
        window_index.reset_index(drop=False, inplace=True)
        window_index.sort_values(by=["time"], inplace=True)

    tbl_optmap = {}
    pg_class = pd.read_csv(f"{input_dir}/pg_class.csv")
    for tup in pg_class.itertuples():
        if tup.reloptions is not None and isinstance(tup.reloptions, str):
            for key, value in re.findall(r'(\w+)=(\w*)', tup.reloptions):
                if key == "fillfactor":
                    tbl_optmap[tup.relname] = float(value)

    Path(f"{input_dir}/train").mkdir(parents=True, exist_ok=True)
    for tbl, keyspaces in table_keyspace_map.items():
        logger.info("Processing table: %s", tbl)

        accum_base_data = []
        key_fn = lambda x: int(x.split(".feather")[0].split("_")[-1])
        files = sorted(glob.glob(f"{input_dir}/windows/{tbl}/queries_*.feather"), key=key_fn)
        data = None

        # FIXME(INDEX): We currently generate the "windows" off of the base table window. This is primarily
        # motivated by the fact that VACUUM only executes on the base table then the indexes (in other words
        # you can't vacuum an index without vacuum'ing the table).
        #
        # If we do want to fully separate out this dependency, then we would need to modify:
        # 1- analyze.py to generate the window_index_map for all the other keyspaces.
        # 2- windowize.py to generate the data ranges for the other keyspaces
        # We don't actually need windowize to split the data files physically, we could just have it write indexes.
        window_data = pd.read_csv(f"{input_dir}/{tbl}.csv")

        tbl_keyspace = []
        if tbl in table_keyspace_map and tbl in table_keyspace_map[tbl]:
            tbl_keyspace = table_keyspace_map[tbl][tbl]

        for query_file in tqdm(files):
            true_window_index = key_fn(query_file)
            query_data = pd.read_feather(query_file)
            if Path(f"{input_dir}/windows/{tbl}/data_{true_window_index}.feather").exists():
                data = pd.read_feather(f"{input_dir}/windows/{tbl}/data_{true_window_index}.feather")

            # Process the base relation.
            window_index_tbl = window_index_map[tbl]
            window_index_slice = window_index_tbl[window_index_tbl.true_window_index == true_window_index]
            assert window_index_slice.shape[0] == 1
            next_window_index = window_index_slice.iloc[0].name + 1
            if next_window_index < window_index_tbl.shape[0] and window_index_tbl.loc[next_window_index].true_window_index != -1:
                next_true_window_index = int(window_index_tbl.loc[next_window_index].true_window_index)
                next_table_tuple = window_data.iloc[next_true_window_index]

                # We pass in a copy of the data in order to prevent mutations from going across data.
                input_data = data.copy() if data is not None else None
                table_tuple = window_data.iloc[true_window_index]
                table_tuple["ff"] = tbl_optmap[tbl]

                point = WorkloadModel.featurize(query_data[query_data.target_index_name.isna()], input_data, table_tuple, tbl_keyspace, train=True, next_table_tuple=next_table_tuple)
                accum_base_data.append(point)

            # FIXME(INDEX): We currently don't have the correct training data. Sigh.
            #for keyspace_name, keyspace in keyspaces:
            #    if keyspace_name == tbl:
            #        continue
            #    logger.info("Processing %s", keyspace_name)

        pd.DataFrame(accum_base_data).to_feather(f"{input_dir}/train/data_{tbl}.feather")
        del accum_base_data


class GenerateTrainCLI(cli.Application):
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
            gen_train(input_parts[i])

if __name__ == "__main__":
    GenerateTrainCLI.run()
