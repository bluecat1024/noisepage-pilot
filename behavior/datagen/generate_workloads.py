from __future__ import annotations

import logging
import shutil
from pathlib import Path

import yaml
from plumbum import cli

from behavior import BENCHDB_TO_TABLES
from evaluation.utils import inject_param_xml, param_sweep_space, parameter_sweep
import knob_change.init_knob_change_log

logger = logging.getLogger(__name__)

def knob_sweep_callback(knobs, closure):
    """
    Callback to knob exploration sweeping.

    For each knob value enumeration, save a separate DBMS configuration file.
    Inject the altered knob values to the DMBS file.
    Return the path name of the configuration file.

    Parameters:
    -----------
    knobs : List[Tuple[List[str], Any]]
        The knob enumeration.
    closure : Dict[str, Any]
        Closure environment passed from caller.

    Returns:
    -----------
    suffixed_dbms_config_file: str
        Resolved path of injected DBMS configuration file.
    """

    results_dir = closure["results_dir"]
    postgresql_config_file = closure["postgresql_config_file"]
    # The knob configuration file name is a concatenation of knobs and values.
    knob_file_name = "_".join([knob_name[0] + "_" + str(value) for knob_name, value in knobs])
    benchbase_postgresql_config_file = Path(results_dir / f"{knob_file_name}.conf")
    shutil.copy(postgresql_config_file, benchbase_postgresql_config_file)
    # Change the content of knobs.
    inject_knob_exploration(benchbase_postgresql_config_file.as_posix(), knobs)

    return benchbase_postgresql_config_file.resolve()


def datagen_sweep_callback(parameters, closure):
    """
    Callback to datagen parameter sweep.

    Given the current set of parameters as part of the sweep, this callback generates
    the correct output workload format. Workload formats are described in more detail
    in behavior/datagen/run_workloads.sh

    Parameters:
    -----------
    parameters: List[Tuple[List[str], Any]]
        The parameter combination.
    closure : Dict[str, Any]
        Closure environment passed from caller.
    """
    mode_dir = closure["mode_dir"]
    benchmark = closure["benchmark"]
    benchbase_config_path = closure["benchbase_config_path"]
    postgresql_config_file = closure["postgresql_config_file"]
    pg_analyze = closure["pg_analyze"]
    pg_prewarm = closure["pg_prewarm"]
    knob_sweep_enabled = closure["knob_sweep_enabled"]
    knob_explore_space = closure["knob_explore_space"]

    # The suffix is a concatenation of parameter names and their values.
    param_suffix = "_".join([name_level[-1] + "_" + str(value) for name_level, value in parameters])
    results_dir = Path(mode_dir / (benchmark + "_" + param_suffix))
    results_dir.mkdir(exist_ok=True)
    print(f"Creating workload configuration: {results_dir}")

    # Copy and inject the XML file of BenchBase.
    benchbase_config_file = Path(results_dir / "benchbase_config.xml")
    shutil.copy(benchbase_config_path, benchbase_config_file)
    inject_param_xml(benchbase_config_file.as_posix(), parameters)
    benchbase_configs = [str(benchbase_config_file.resolve())]

    # Copy the default postgresql.conf file.
    # TODO(wz2): Rewrite the postgresql.conf based on knob tweaks and modify the param_suffix above.
    # If knob sweeping is enabled, generate a list of different configuration files.
    if knob_sweep_enabled:
        knob_sweep_closure = {
            "postgresql_config_file": postgresql_config_file,
            "results_dir": results_dir,
        }
        pg_configs = parameter_sweep(knob_explore_space, knob_sweep_callback, knob_sweep_closure)
        # The same benchbase configuration will be used.
        benchbase_configs = benchbase_configs * len(pg_configs)
    else:
        benchbase_postgresql_config_file = Path(results_dir / "postgresql.conf")
        shutil.copy(postgresql_config_file, benchbase_postgresql_config_file)
        pg_configs = [str(benchbase_postgresql_config_file.resolve())]

    # create the separate file to record global knob changes.
    knob_change_log = Path(results_dir / "benchbase_config.xml").as_posix()
    init_knob_change_log(knob_change_log, knob_explore_space)
    # Create the config.yaml file
    config = {
        "benchmark": benchmark,
        "pg_analyze": pg_analyze,
        "pg_prewarm": pg_prewarm,
        "pg_configs": pg_configs,
        "benchbase_configs": benchbase_configs,
        "knob_change_log": knob_change_log,
    }

    with (results_dir / "config.yaml").open("w") as f:
        yaml.dump(config, f)


class GenerateWorkloadsCLI(cli.Application):
    config_file = cli.SwitchAttr(
        "--config-file",
        Path,
        mandatory=True,
        help="Path to configuration YAML containing datagen parameters.",
    )
    postgresql_config_file = cli.SwitchAttr(
        "--postgresql-config-file",
        Path,
        mandatory=True,
        help="Path to standard postgresql.conf that the workloads should execute with.",
    )
    dir_benchbase_config = cli.SwitchAttr(
        "--dir-benchbase-config",
        Path,
        mandatory=True,
        help="Path to BenchBase config files.",
    )
    dir_output = cli.SwitchAttr(
        "--dir-output",
        Path,
        mandatory=True,
        help="Directory to write generated data to.",
    )

    def main(self):
        config_path = Path(self.config_file)
        with config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)["datagen"]
        logger.setLevel(self.config["log_level"])

        # Validate the chosen benchmarks from the config.
        benchmarks = self.config["benchmarks"]
        for benchmark in benchmarks:
            if benchmark not in BENCHDB_TO_TABLES:
                raise ValueError(f"Invalid benchmark: {benchmark}")

        self.dir_output.mkdir(parents=True, exist_ok=True)
        modes = ["train", "eval"]
        for mode in modes:
            mode_dir = Path(self.dir_output) / mode
            Path(mode_dir).mkdir(parents=True, exist_ok=True)

            # Build sweeping space
            ps_space = param_sweep_space(self.config["param_sweep"])
            knob_explore_space = param_sweep_space(self.config["knob_sweep"])
            # For each benchmark, ...
            for benchmark in benchmarks:
                benchbase_config_path = self.dir_benchbase_config / f"{benchmark}_config.xml"

                # Generate OU training data for every parameter combination.
                closure = {
                    "mode_dir": mode_dir,
                    "benchmark": benchmark,
                    "benchbase_config_path": benchbase_config_path,
                    "postgresql_config_file": self.postgresql_config_file,
                    "pg_prewarm": self.config["pg_prewarm"],
                    "pg_analyze": self.config["pg_analyze"],
                    "knob_sweep_enabled": self.config["knob_sweep_enabled"],
                    "knob_explore_space": knob_explore_space,
                }
                parameter_sweep(ps_space, datagen_sweep_callback, closure)


if __name__ == "__main__":
    GenerateWorkloadsCLI.run()
