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

    def start_collector(output_dir, wait_time, collector_fast_interval, collector_slow_interval):
        if output_dir is None:
            print("Unable to start collector without an output directory")
            return False

        output_dir = Path(output_dir).absolute()
        print("Attaching collector.")

        dir_collector = BUILD_PATH / "cmudb/collector"
        os.chdir(dir_collector)

        arguments = [
            "collector.py",
            "--outdir",
            output_dir,
            "--collector_fast_interval",
            collector_fast_interval,
            "--collector_slow_interval",
            collector_slow_interval
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
                "name": "collector_fast_interval",
                "long": "collector_fast_interval",
                "help": "Interval (seconds) to collect (frequent) information from database.",
                "default": 1,
            },
            {
                "name": "collector_slow_interval",
                "long": "collector_slow_interval",
                "help": "Interval (seconds) to collect (infrequent) information from database.",
                "default": 60,
            },
        ],
    }


def task_collector_shutdown():
    """
    Collector: shutdown the running collector instance.
    """

    def shutdown_collector():
        local["pkill"]["-SIGINT", "-i", "-f", "collector.py"](retcode=None)
        while len(list(local.pgrep("collector.py"))) != 0:
            print("Waiting for collector to shutdown from SIGINT.")
            time.sleep(5)

        # Nuke that irritating psycache folder.
        dir_collector = BUILD_PATH / "cmudb/collector/__pycache__"
        cmd.sudo["rm", "-rf", dir_collector]()

    return {
        "actions": [shutdown_collector],
        "file_dep": [ARTIFACT_pg_ctl],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
    }
