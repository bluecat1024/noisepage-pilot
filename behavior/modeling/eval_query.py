import logging
import glob
import shutil
import gc
import re
import psycopg
import copy
from psycopg.rows import dict_row
from distutils import util
import json
from datetime import datetime
import numpy as np
import itertools
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from plumbum import cli

from behavior import OperatingUnit, Targets, BENCHDB_TO_TABLES
from behavior.modeling.utils.evaluate_ou import evaluate_ou_model
from behavior.modeling.utils.postgres import prepare_augmentation_data, prepare_pg_inference_state
from behavior.modeling.utils.prepare_data import purify_index_input_data, prepare_inference_query_stream
from behavior.modeling.utils.postgres_model_plan import generate_vacuum_partition, generate_query_ous, estimate_query_modifications
from behavior.plans.utils import (
    process_time_pg_stats,
    process_time_pg_attribute,
    process_time_pg_index,
    process_time_pg_class,
    merge_modifytable_data,
    build_time_index_metadata
)

logger = logging.getLogger(__name__)


def load_models(path):
    model_dict = {}
    for model_path in path.rglob('*.pkl'):
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        model_dict[model.ou_name] = model
    return model_dict


def augment_ous(scratch_it, sliced_metadata, conn):
    # Prepare all the augmented catalog data in timestamp order.
    aug_dfs = prepare_augmentation_data(sliced_metadata, conn)
    process_tables, process_idxs = process_time_pg_class(aug_dfs["pg_class"])
    process_pg_index = process_time_pg_index(aug_dfs["pg_index"])
    process_pg_attribute = process_time_pg_attribute(aug_dfs["pg_attribute"])
    time_pg_index = build_time_index_metadata(process_pg_index, process_tables.copy(deep=True), process_idxs, process_pg_attribute)

    time_pg_stats = process_time_pg_stats(aug_dfs["pg_stats"])
    time_pg_stats.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    time_pg_stats.sort_index(axis=0, inplace=True)

    pg_settings = aug_dfs["pg_settings"]
    pg_settings["unix_timestamp"] = pg_settings.unix_timestamp.astype(np.float64)
    pg_settings.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    pg_settings.sort_index(axis=0, inplace=True)

    index_augment = [OperatingUnit.IndexScan, OperatingUnit.IndexOnlyScan, OperatingUnit.ModifyTableIndexInsert]
    for augment in index_augment:
        column = {
            OperatingUnit.IndexOnlyScan: "IndexOnlyScan_indexid",
            OperatingUnit.IndexScan: "IndexScan_indexid",
            OperatingUnit.ModifyTableIndexInsert: "ModifyTableIndexInsert_indexid"
        }[augment]

        files = sorted(glob.glob(f"{scratch_it}/{augment.name}.feather.*"))
        for target_file in files:
            if target_file.startswith("AUG"):
                continue

            logger.info("[AUGMENT] Input %s", target_file)
            data = pd.read_feather(target_file)
            data.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
            data.sort_index(axis=0, inplace=True)

            data = pd.merge_asof(data, pg_settings, left_index=True, right_index=True, allow_exact_matches=True)
            data = pd.merge_asof(data, time_pg_index, left_index=True, right_index=True, left_by=[column], right_by=["indexrelid"], allow_exact_matches=True)
            assert data.indexrelid.isna().sum() == 0

            indkey_atts = [key for key in data.columns if "indkey_attname_" in key]
            for idx, indkey_att in enumerate(indkey_atts):
                left_by = ["table_relname", indkey_att]
                right_by = ["tablename", "attname"]
                data = pd.merge_asof(data, time_pg_stats, left_index=True, right_index=True, left_by=left_by, right_by=right_by, allow_exact_matches=True)

                # Rename the key and drop the other useless columns.
                data.drop(labels=["tablename", "attname"], axis=1, inplace=True)
                remapper = {column:f"indkey_{column}_{idx}" for column in time_pg_stats.columns}
                data.rename(columns=remapper, inplace=True)

            data = purify_index_input_data(data)
            data.reset_index(drop=True, inplace=True)
            data.to_feather(scratch_it / f"AUG_{augment.name}.feather{Path(target_file).suffix}")

    mt_augment = [OperatingUnit.ModifyTableInsert, OperatingUnit.ModifyTableUpdate]
    process_tables.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    process_tables.sort_index(axis=0, inplace=True)
    for augment in mt_augment:
        files = sorted(glob.glob(f"{scratch_it}/{augment.name}.feather.*"))
        for target_file in files:
            logger.info("[AUGMENT] Input %s", target_file)
            data = pd.read_feather(target_file)
            data = merge_modifytable_data(data=data, processed_time_tables=process_tables)
            data.to_feather(scratch_it / f"AUG_{augment.name}.feather{Path(target_file).suffix}")

    open(scratch_it / "augment_query_ous", "w").close()


def evaluate_query_plans(split_query_stream, base_models, scratch_it, output_path):
    # Evaluate all the OUs.
    for ou in OperatingUnit:
        ou_type = ou.name
        files = sorted(glob.glob(f"{scratch_it}/AUG_{ou.name}.feather.*"))
        def logged_read_feather(target):
            logger.info("[^]: [LOAD] Input %s", target)
            return pd.read_feather(target)

        if len(files) > 0:
            df = pd.concat(map(logged_read_feather, files))
            df.reset_index(drop=True, inplace=True)
        else:
            files = sorted(glob.glob(f"{scratch_it}/{ou.name}.feather.*"))
            if len(files) > 0:
                df = pd.concat(map(logged_read_feather, files))
                df.reset_index(drop=True, inplace=True)
            else:
                # This OU has no data.
                continue

        if ou_type not in base_models:
            # If we don't have the model for the particular OU, we just predict 0.
            df["pred_elapsed_us"] = 0
            # Set a bit in [error_missing_model]
            df["error_missing_model"] = 1
        else:
            df = evaluate_ou_model(base_models[ou_type], None, None, eval_df=df, return_df=True, output=False)
            df["error_missing_model"] = 0

        if OperatingUnit[ou_type] == OperatingUnit.IndexOnlyScan or OperatingUnit[ou_type] == OperatingUnit.IndexScan:
            # TODO(wz2): Assume each other loop executes with the same latency.
            prefix = "IndexOnlyScan" if OperatingUnit[ou_type] == OperatingUnit.IndexOnlyScan else "IndexScan"
            df["pred_elapsed_us"] = df.pred_elapsed_us * df[f"{prefix}_num_outer_loops"]

        df.to_feather(output_path / f"{ou_type}_evals.feather")

    # Massage the frames together from disk to combat OOM.
    unified_df = None
    for ou in OperatingUnit:
        ou_type = ou.name
        if (output_path / f"{ou_type}_evals.feather").exists():
            logger.info("Reading %s", ou_type)
            df = pd.read_feather(output_path / f"{ou_type}_evals.feather")
            keep_columns = ["query_id", "order", "pred_elapsed_us", "error_missing_model"]
            df.drop(columns=[col for col in df.columns if col not in keep_columns], inplace=True)
            if unified_df is None:
                unified_df = df
            else:
                unified_df = pd.concat([unified_df, df], ignore_index=True).groupby(["query_id", "order"]).sum()
                unified_df.reset_index(drop=False, inplace=True)
            gc.collect()

    assert unified_df is not None
    unified_df.drop(columns=["query_id"], inplace=True)
    unified_df.set_index(keys=["order"], drop=True, append=False, inplace=True)

    reconstitute_stream = pd.concat(split_query_stream)
    reconstitute_stream.drop(columns=["pred_elapsed_us", "error_missing_model"], inplace=True, errors='ignore')
    reconstitute_stream.set_index(keys=["order"], drop=True, append=False, inplace=True)
    reconstitute_stream.sort_index(axis=0, inplace=True)

    reconstitute_stream = reconstitute_stream.join(unified_df, how="inner")
    reconstitute_stream.reset_index(drop=False, inplace=True)
    return reconstitute_stream


def _process_queries(dir_data, scratch):
    if not (scratch / "raw_queries.feather").exists():
        # Get the settings and then get the initial state of tables.
        query_stream = prepare_inference_query_stream(dir_data)
        query_stream.to_feather(scratch / "raw_queries.feather")
        # Force a reload to contract memory usage.
        del query_stream

    return pd.read_feather(scratch / "raw_queries.feather")


def _slice_queries(query_stream, conn, scratch_it, iteration_count):
    if not (scratch_it / "sliced_query_parse.pickle").exists():
        metadata = prepare_pg_inference_state(conn)
        query_stream, metadata = estimate_query_modifications(query_stream, metadata, skip_query=iteration_count != 0)
        sliced_query_stream, sliced_metadata = generate_vacuum_partition(query_stream, metadata)

        [df.to_feather(f"{scratch_it}/sliced_query_stream.feather.{i}") for i, df in enumerate(sliced_query_stream)]
        with open(scratch_it / "sliced_query_parse.pickle", "wb+") as f:
            pickle.dump(sliced_metadata, f)

    qs_files = sorted(glob.glob(f"{scratch_it}/sliced_query_stream.feather.*"))
    sliced_query_stream = [pd.read_feather(f) for f in qs_files]
    with open(scratch_it / "sliced_query_parse.pickle", "rb") as f:
        sliced_metadata = pickle.load(f)
    return sliced_query_stream, sliced_metadata


def main(num_iterations, psycopg2_conn, session_sql, dir_models, dir_data, dir_evals_output, dir_scratch, predictive):
    logger.info("Executing in predictive mode: %s", predictive)
    if not predictive:
        # Only run for 1 iteration if not in predictive mode.
        num_iterations = 1

    # Load the models
    base_models = load_models(dir_models)

    eval_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output = dir_evals_output / f"eval_{eval_timestamp}"

    # Scratch space is used to try and reduce debugging overhead.
    scratch = dir_scratch / "eval_query_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    iteration_count = 0
    with psycopg.connect(psycopg2_conn, autocommit=True) as conn:
        conn.execute("SET qss_capture_enabled = OFF")
        if session_sql.exists():
            with open(session_sql, "r") as f:
                for line in f:
                    conn.execute(line)

        while iteration_count < num_iterations:
            scratch_it = scratch / f"{iteration_count}"
            scratch_it.mkdir(parents=True, exist_ok=True)
            if (scratch_it / "final_query.feather").exists():
                # Already finished this iteration so move to the next one.
                if "query_stream" in locals():
                    del query_stream
                    gc.collect()

                query_stream = pd.read_feather(scratch_it / "final_query.feather")
                iteration_count = iteration_count + 1
                continue
            elif iteration_count == 0:
                logger.info("[%s, %s]: Processing queries", iteration_count, datetime.now())
                query_stream = _process_queries(dir_data, scratch)

            logger.info("[%s, %s]: Slicing query stream based on vacuum", iteration_count, datetime.now())
            sliced_query_stream, sliced_metadata = _slice_queries(query_stream, conn, scratch_it, iteration_count)
            del query_stream
            gc.collect()
            logger.info("[%s, %s]: Produced [%s] query streams based on vacuum analysis", iteration_count, datetime.now(), len(sliced_query_stream))

            logger.info("[%s, %s]: Generate query operating units", iteration_count, datetime.now())
            if not (scratch_it / "generate_query_ous").exists():
                generate_query_ous(sliced_query_stream, sliced_metadata, conn, scratch_it, predictive)
            gc.collect()

            logger.info("[%s, %s]: Augmenting query operating units", iteration_count, datetime.now())
            if not (scratch_it / "augment_query_ous").exists():
                augment_ous(scratch_it, sliced_metadata, conn)

            # No longer need sliced_metadata after this point.
            del sliced_metadata
            gc.collect()

            target_output = base_output / f"preds_{iteration_count}"
            target_output.mkdir(parents=True, exist_ok=True)

            # Evaluate query plans and relase resources.
            logger.info("[%s, %s]: Evaluating query plans", iteration_count, datetime.now())
            query_stream = evaluate_query_plans(sliced_query_stream, base_models, scratch_it, target_output)
            del sliced_query_stream
            gc.collect()

            # Generate prediction plots.
            query_stream.to_feather(target_output / "query_results.feather")
            query_stream.to_feather(scratch_it / "final_query.feather")

            iteration_count = iteration_count + 1

    # Remove the scratch space.
    shutil.rmtree(scratch)


class EvalQueryCLI(cli.Application):
    session_sql = cli.SwitchAttr(
        "--session-sql",
        Path,
        mandatory=False,
        help="Path to a list of SQL statements that should be executed in the session prior to EXPLAIN.",
    )
    dir_data = cli.SwitchAttr(
        "--dir-data",
        Path,
        mandatory=True,
        help="Folder containing raw evaluation CSVs.",
    )
    dir_evals_output = cli.SwitchAttr(
        "--dir-evals-output",
        Path,
        mandatory=True,
        help="Folder to output evaluations to.",
    )
    dir_scratch = cli.SwitchAttr(
        "--dir-scratch",
        Path,
        mandatory=False,
        help="Folder to use as scratch space.",
        default = Path("/tmp/"),
    )
    dir_models = cli.SwitchAttr(
        "--dir-base-models",
        Path,
        mandatory=True,
        help="Folder containing the base evaluation models.",
    )
    psycopg2_conn = cli.SwitchAttr(
        "--psycopg2-conn",
        mandatory=True,
        help="Psycopg2 connection string for connecting to a valid database.",
    )
    num_iterations = cli.SwitchAttr(
        "--num-iterations",
        int,
        mandatory=False,
        help="Number of iterations to attempt to converge predictions.",
    )
    predictive = cli.SwitchAttr(
        "--predictive",
        mandatory=False,
        help="Whether to enable predictive mode or not.",
    )

    def main(self):
        main(self.num_iterations,
             self.psycopg2_conn,
             self.session_sql,
             self.dir_models,
             self.dir_data,
             self.dir_evals_output,
             self.dir_scratch,
             self.predictive == "True")


if __name__ == "__main__":
    EvalQueryCLI.run()
