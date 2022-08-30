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

logger = logging.getLogger("gen_train")


def gen_train(input_dir, output_dir, benchmark):
    TABLES = BENCHDB_TO_TABLES[benchmark]
    with open(f"{input_dir}/analysis/keyspaces.pickle", "rb") as f:
        table_attr_map = pickle.load(f)
        attr_table_map = pickle.load(f)
        table_keyspace_map = pickle.load(f)
        query_template_map = pickle.load(f)
        window_index_map = pickle.load(f)

    pg_class = pd.read_csv(f"{input_dir}/pg_class.csv")
    tbl_optmap = {}
    for tup in pg_class.itertuples():
        if tup.reloptions is not None and isinstance(tup.reloptions, str):
            for key, value in re.findall(r'(\w+)=(\w*)', tup.reloptions):
                if key == "fillfactor":
                    tbl_optmap[tup.relname] = float(value)

    Path(f"{output_dir}/train").mkdir(parents=True, exist_ok=True)
    for tbl in TABLES:
        logger.info("Processing table: %s", tbl)
        accum_data = []
        key_fn = lambda x: int(x.split(".feather")[0].split("_")[-1])
        files = sorted(glob.glob(f"{input_dir}/windows/{tbl}/queries_*.feather"), key=key_fn)
        if Path(f"{input_dir}/windows/{tbl}/dataat_0.feather").exists():
            data = pd.read_feather(f"{input_dir}/windows/{tbl}/dataat_0.feather")
        else:
            data = None

        window_data = pd.read_csv(f"{input_dir}/{tbl}.csv")
        for query_file in tqdm(files):
            query_data = pd.read_feather(query_file)
            if Path(f"{input_dir}/windows/{tbl}/dataat_{key_fn(query_file)}.feather").exists():
                data = pd.read_feather(f"{input_dir}/windows/{tbl}/dataat_{key_fn(query_file)}.feather")

            keyspace = []
            if tbl in table_keyspace_map and tbl in table_keyspace_map[tbl]:
                keyspace = table_keyspace_map[tbl][tbl]

            # We pass in a copy of the data in order to prevent mutations from going across data.
            input_data = data.copy() if data is not None else None
            table_tuple = window_data.iloc[key_fn(query_file)]
            table_tuple["ff"] = tbl_optmap[tbl]

            point = WorkloadModel.featurize(query_data, input_data, table_tuple, keyspace, train=True)
            accum_data.append(point)

        pd.DataFrame(accum_data).to_feather(f"{output_dir}/train/data_{tbl}.feather")


class GenerateTrainCLI(cli.Application):
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
            gen_train(input_parts[i], output_parts[i], b_parts[i])

if __name__ == "__main__":
    GenerateTrainCLI.run()
