import os
import sys
import time
from pathlib import Path

from plumbum import cmd, local

from dodos import VERBOSITY_DEFAULT
from dodos.noisepage import BUILD_PATH, ARTIFACT_pg_ctl


def task_collector_init():
    """
    Collector: attach collector to a running NoisePage instance.
    """

    def start_collector(benchmark, output_dir, wait_time, collector_interval):
        if output_dir is None:
            print("Unable to start collector without an output directory")
            return False

        if benchmark is None:
            print("Unable to start collector without a benchmark")
            return False

        output_dir = Path(output_dir).absolute()
        print("Attaching collector.")

        arguments = [
            "-m",
            "behavior",
            "collector",
            "--benchmark",
            benchmark,
            "--outdir",
            output_dir,
            "--collector_interval",
            collector_interval
        ]

        local["python3"][arguments].run_bg(
            # sys.stdout will actually give the doit writer. Here we need the actual
            # underlying output stream.
            stdout=sys.__stdout__,
            stderr=sys.__stderr__,
        )

        time.sleep(int(wait_time))

    return {
        "actions": [start_collector],
        "file_dep": [ARTIFACT_pg_ctl],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "benchmark",
                "long": "benchmark",
                "help": "Benchmark name that is executed",
                "default": None,
            },
            {
                "name": "output_dir",
                "long": "output_dir",
                "help": "Directory that collector should output to",
                "default": None,
            },
            {
                "name": "wait_time",
                "long": "wait_time",
                "help": "Time to wait (seconds) after collector has been started.",
                "default": 10,
            },
            {
                "name": "collector_interval",
                "long": "collector_interval",
                "help": "Interval (seconds) to collect (infrequent) information from database.",
                "default": 30,
            },
        ],
    }


def task_collector_shutdown():
    """
    Collector: shutdown the running collector instance.
    """

    def shutdown_collector():
        local["pkill"]["-SIGINT", "-i", "-f", "behavior collector"](retcode=None)
        local["pkill"]["-SIGINT", "-i", "-f", "Userspace Collector"](retcode=None)
        while len(list(local.pgrep("behavior collector"))) != 0:
            print("Waiting for collector to shutdown from SIGINT.")
            time.sleep(5)

        print("Shutdown collector with SIGINT.")

    return {
        "actions": [shutdown_collector],
        "file_dep": [ARTIFACT_pg_ctl],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
    }
