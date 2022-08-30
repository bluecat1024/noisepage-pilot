import os
from pathlib import Path

from doit.action import CmdAction
from plumbum import local, FG

import dodos.benchbase
import dodos.noisepage
from dodos import VERBOSITY_DEFAULT, default_artifacts_path, default_build_path
from dodos.benchbase import ARTIFACTS_PATH as BENCHBASE_ARTIFACTS_PATH
from dodos.noisepage import (
    ARTIFACTS_PATH as NOISEPAGE_ARTIFACTS_PATH,
    ARTIFACT_pgdata,
    ARTIFACT_psql,
)
from behavior import BENCHDB_TO_TABLES

ARTIFACTS_PATH = default_artifacts_path()
ARTIFACT_MODELS = ARTIFACTS_PATH / "workload_models"
BUILD_PATH = default_build_path()


def task_workload_analyze():
    """
    Workload Analysis: perform analysis of a workload
    """
    def workload_analyze(benchmark, input_workload, output_workload, workload_only, psycopg2_conn):
        assert input_workload is not None and output_workload is not None
        assert len(benchmark.split(",")) == len(input_workload.split(",")) == len(output_workload.split(","))

        for iw in input_workload.split(","):
            assert Path(iw).exists(), f"{iw} is not valid path."

        for iw in output_workload.split(","):
            assert Path(iw).exists(), f"{iw} is not valid path."

        for bw in benchmark.split(","):
            assert bw in BENCHDB_TO_TABLES

        eval_args = (
            f"--benchmark {benchmark} "
            f"--dir-workload-input {input_workload} "
            f"--dir-workload-output {output_workload} "
            f"--workload-only {workload_only} "
        )

        if psycopg2_conn is not None:
            eval_args = eval_args + f"--psycopg2-conn \"{psycopg2_conn}\" "

        return f"python3 -m behavior workload_analyze {eval_args}"

    return {
        "actions": [CmdAction(workload_analyze, buffering=1),],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "benchmark",
                "long": "benchmark",
                "help": "Benchmark that is being analyzed.",
                "default": None,
            },
            {
                "name": "input_workload",
                "long": "input_workload",
                "help": "Path to the input workload that should be analyzed.",
                "default": None,
            },
            {
                "name": "output_workload",
                "long": "output_workload",
                "help": "Path to the output workload that should be analyzed.",
                "default": None,
            },
            {
                "name": "workload_only",
                "long": "workload_only",
                "help": "Whether the input workload is only the workload or not.",
                "default": False,
            },
            {
                "name": "psycopg2_conn",
                "long": "psycopg2_conn",
                "help": "psycopg2 connection string to connect to the valid database instance.",
                "default": None,
            },
        ],
    }


def task_workload_populate_data():
    """
    Workload Analysis: populate missing parameters of a workload and infer execution characteristics (if needed).
    """
    def workload_populate_data(benchmark, input_workload, output_workload, workload_only, psycopg2_conn):
        assert input_workload is not None and output_workload is not None
        assert len(benchmark.split(",")) == len(input_workload.split(",")) == len(output_workload.split(","))

        for iw in input_workload.split(","):
            assert Path(iw).exists(), f"{iw} is not valid path."

        for iw in output_workload.split(","):
            assert Path(iw).exists(), f"{iw} is not valid path."

        for bw in benchmark.split(","):
            assert bw in BENCHDB_TO_TABLES

        eval_args = (
            f"--benchmark {benchmark} "
            f"--dir-workload-input {input_workload} "
            f"--dir-workload-output {output_workload} "
            f"--workload-only {workload_only} "
        )

        if psycopg2_conn is not None:
            eval_args = eval_args + f"--psycopg2-conn \"{psycopg2_conn}\" "

        return f"python3 -m behavior workload_populate_data {eval_args}"

    return {
        "actions": [CmdAction(workload_populate_data, buffering=1),],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "benchmark",
                "long": "benchmark",
                "help": "Benchmark that is being analyzed.",
                "default": None,
            },
            {
                "name": "input_workload",
                "long": "input_workload",
                "help": "Path to the input workload that should be analyzed.",
                "default": None,
            },
            {
                "name": "output_workload",
                "long": "output_workload",
                "help": "Path to the output workload that should be analyzed.",
                "default": None,
            },
            {
                "name": "workload_only",
                "long": "workload_only",
                "help": "Whether the input workload is only the workload or not.",
                "default": False,
            },
            {
                "name": "psycopg2_conn",
                "long": "psycopg2_conn",
                "help": "psycopg2 connection string to connect to the valid database instance.",
                "default": None,
            },
        ],
    }


def task_workload_windowize():
    """
    Workload Analysis: construct workload windows based on sampling.
    """
    def windowize(benchmark, input_workload, output_workload):
        assert input_workload is not None and output_workload is not None
        assert len(benchmark.split(",")) == len(input_workload.split(",")) == len(output_workload.split(","))

        for iw in input_workload.split(","):
            assert Path(iw).exists(), f"{iw} is not valid path."

        for iw in output_workload.split(","):
            assert Path(iw).exists(), f"{iw} is not valid path."

        for bw in benchmark.split(","):
            assert bw in BENCHDB_TO_TABLES

        eval_args = (
            f"--benchmark {benchmark} "
            f"--dir-workload-input {input_workload} "
            f"--dir-workload-output {output_workload} "
        )

        return f"python3 -m behavior workload_windowize {eval_args}"

    return {
        "actions": [CmdAction(windowize, buffering=1),],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "benchmark",
                "long": "benchmark",
                "help": "Benchmark that is being analyzed.",
                "default": None,
            },
            {
                "name": "input_workload",
                "long": "input_workload",
                "help": "Path to the input workload that should be analyzed.",
                "default": None,
            },
            {
                "name": "output_workload",
                "long": "output_workload",
                "help": "Path to the output workload that should be analyzed.",
                "default": None,
            },
        ],
    }


def task_workload_prepare_train():
    """
    Workload Analysis: construct training data from windows.
    """
    def prepare_train(benchmark, input_workload, output_workload):
        assert input_workload is not None and output_workload is not None
        assert len(benchmark.split(",")) == len(input_workload.split(",")) == len(output_workload.split(","))

        for iw in input_workload.split(","):
            assert Path(iw).exists(), f"{iw} is not valid path."

        for iw in output_workload.split(","):
            assert Path(iw).exists(), f"{iw} is not valid path."

        for bw in benchmark.split(","):
            assert bw in BENCHDB_TO_TABLES

        eval_args = (
            f"--benchmark {benchmark} "
            f"--dir-workload-input {input_workload} "
            f"--dir-workload-output {output_workload} "
        )

        return f"python3 -m behavior workload_prepare_train {eval_args}"

    return {
        "actions": [CmdAction(prepare_train, buffering=1),],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "benchmark",
                "long": "benchmark",
                "help": "Benchmark that is being analyzed.",
                "default": None,
            },
            {
                "name": "input_workload",
                "long": "input_workload",
                "help": "Path to the input workload that should be analyzed.",
                "default": None,
            },
            {
                "name": "output_workload",
                "long": "output_workload",
                "help": "Path to the output workload that should be analyzed.",
                "default": None,
            },
        ],
    }


def task_workload_train():
    """
    Workload Analysis: train workload models.
    """

    def train_cmd(input_data, output_dir, separate, val_size, epochs, batch_size, hidden_size):
        if not Path(output_dir).is_absolute():
            # Make it a relative path to ARTIFACT_MODELS.
            output_dir = ARTIFACT_MODELS / output_dir

        train_args = (
            f"--dir-input \"{input_data}\" "
            f"--dir-output {output_dir} "
            f"--separate {separate} "
            f"--val-size {val_size} "
            f"--epochs {epochs} "
            f"--batch-size {batch_size} "
            f"--hidden-size {hidden_size} "
        )

        return f"python3 -m behavior workload_train {train_args}"

    return {
        "actions": [f"mkdir -p {ARTIFACT_MODELS}", CmdAction(train_cmd, buffering=1)],
        "targets": [ARTIFACT_MODELS],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
        "params": [
            {
                "name": "input_data",
                "long": "input_data",
                "help": "Path to all the input data that should be used, comma separated if multiple globs.",
                "default": None,
            },
            {
                "name": "output_dir",
                "long": "output_dir",
                "help": "Path to the output folder where the workload models should be written to.",
                "default": None,
            },
            {
                "name": "separate",
                "long": "separate",
                "help": "Whether to train separate models for each workload execution feature.",
                "default": False,
            },
            {
                "name": "val_size",
                "long": "val_size",
                "help": "Percentage of the validation dataset.",
                "type": float,
                "default": 0.2,
            },
            {
                "name": "epochs",
                "long": "epochs",
                "help": "Number of epochs to train the model.",
                "type": int,
                "default": 1000,
            },
            {
                "name": "batch_size",
                "long": "batch_size",
                "help": "Batch size that should be used.",
                "type": int,
                "default": 1024,
            },
            {
                "name": "hidden_size",
                "long": "hidden_size",
                "help": "Number of hidden units.",
                "type": int,
                "default": 256,
            },
        ],
    }


def task_workload_eval():
    """
    Workload Analysis: evaluate workload models.
    """

    def eval_cmd(input_data, model_input_dir, batch_size):
        eval_args = (
            f"--dir-input \"{input_data}\" "
            f"--model-input {model_input_dir} "
            f"--batch-size {batch_size} "
        )

        return f"python3 -m behavior workload_eval {eval_args}"

    return {
        "actions": [f"mkdir -p {ARTIFACT_MODELS}", CmdAction(eval_cmd, buffering=1)],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
        "params": [
            {
                "name": "input_data",
                "long": "input_data",
                "help": "Path to all the input data that should be used, comma separated if multiple globs.",
                "default": None,
            },
            {
                "name": "model_input_dir",
                "long": "model_input_dir",
                "help": "Path to the input folder containing the model.",
                "default": None,
            },
            {
                "name": "batch_size",
                "long": "batch_size",
                "help": "Batch size that should be used.",
                "type": int,
                "default": 1024,
            },
        ],
    }
