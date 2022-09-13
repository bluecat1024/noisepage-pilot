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
    def workload_analyze(benchmark, input_workload, workload_only, psycopg2_conn, slice_window):
        assert input_workload is not None
        assert len(benchmark.split(",")) == len(input_workload.split(","))

        for iw in input_workload.split(","):
            assert Path(iw).exists(), f"{iw} is not valid path."

        for bw in benchmark.split(","):
            assert bw in BENCHDB_TO_TABLES

        eval_args = (
            f"--benchmark {benchmark} "
            f"--dir-workload-input {input_workload} "
            f"--workload-only {workload_only} "
            f"--slice-window {slice_window} "
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
            {
                "name": "slice_window",
                "long": "slice_window",
                "help": "Slice of the window that should be processed.",
                "default": None,
            },
        ],
    }


def task_workload_populate_data():
    """
    Workload Analysis: populate missing parameters of a workload and infer execution characteristics (if needed).
    """
    def workload_populate_data(input_workload, workload_only, psycopg2_conn, slice_window, skip_save_frames):
        assert input_workload is not None

        for iw in input_workload.split(","):
            assert Path(iw).exists(), f"{iw} is not valid path."

        eval_args = (
            f"--dir-workload-input {input_workload} "
            f"--workload-only {workload_only} "
            f"--slice-window {slice_window} "
        )

        if skip_save_frames is not None:
            eval_args += f"--skip-save-frames {skip_save_frames} "

        if psycopg2_conn is not None:
            eval_args = eval_args + f"--psycopg2-conn \"{psycopg2_conn}\" "

        return f"python3 -m behavior workload_populate_data {eval_args}"

    return {
        "actions": [CmdAction(workload_populate_data, buffering=1),],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "input_workload",
                "long": "input_workload",
                "help": "Path to the input workload that should be analyzed.",
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
            {
                "name": "slice_window",
                "long": "slice_window",
                "help": "Slice of the window that should be processed.",
                "default": None,
            },
            {
                "name": "skip_save_frames",
                "long": "skip_save_frames",
                "help": "Whether to save intermediate data frames or not.",
                "default": None,
            },
        ],
    }


def task_workload_windowize():
    """
    Workload Analysis: construct workload windows based on sampling.
    """
    def windowize(input_workload):
        assert input_workload is not None

        for iw in input_workload.split(","):
            assert Path(iw).exists(), f"{iw} is not valid path."

        eval_args = (
            f"--dir-workload-input {input_workload} "
        )

        return f"python3 -m behavior workload_windowize {eval_args}"

    return {
        "actions": [CmdAction(windowize, buffering=1),],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "input_workload",
                "long": "input_workload",
                "help": "Path to the input workload that should be analyzed.",
                "default": None,
            },
        ],
    }


def task_workload_prepare_train():
    """
    Workload Analysis: construct training data from windows.
    """
    def prepare_train(input_workload, hist_length):
        assert input_workload is not None

        for iw in input_workload.split(","):
            assert Path(iw).exists(), f"{iw} is not valid path."

        eval_args = (
            f"--dir-workload-input {input_workload} "
            f"--hist-length {hist_length} "
        )

        return f"python3 -m behavior workload_prepare_train {eval_args}"

    return {
        "actions": [CmdAction(prepare_train, buffering=1),],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "input_workload",
                "long": "input_workload",
                "help": "Path to the input workload that should be analyzed.",
                "default": None,
            },
            {
                "name": "hist_length",
                "long": "hist_length",
                "help": "Length of histogram featurization to use.",
                "default": 10,
            },
        ],
    }


def task_workload_train():
    """
    Workload Analysis: train workload models.
    """

    def train_cmd(input_data, output_dir, separate, val_size, lr, epochs, batch_size, hidden_size, hist_length, cuda):
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
            f"--hist-length {hist_length} "
        )

        if cuda is not None:
            train_args += f"--cuda "

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
                "name": "lr",
                "long": "lr",
                "help": "Learning rate for Adam optimizer.",
                "type": float,
                "default": 0.001,
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
            {
                "name": "hist_length",
                "long": "hist_length",
                "help": "Length of histogram featurization.",
                "type": int,
                "default": 10,
            },
            {
                "name": "cuda",
                "long": "cuda",
                "help": "Whether to use CUDA or not.",
                "default": None,
            },
        ],
    }
