from __future__ import annotations

import os
import xml.etree.ElementTree as ET
import logging
import shutil
from pathlib import Path

import yaml
from plumbum import cli

logger = logging.getLogger(__name__)


class SkynetCLI(cli.Application):
    config_file = cli.SwitchAttr(
        "--config-file",
        Path,
        mandatory=True,
        help="Path to configuration YAML containing datagen parameters.",
    )

    def output_generate(self, run):
        run.write("\n")
        run.write("#############################\n")
        run.write("# Collection and Process\n")
        run.write("#############################\n")
        run.write(("#" if not self.config["workload_generate"] else "") + "doit behavior_generate_workloads\n")
        run.write(("#" if not self.config["workload_execute"] else "") + "rm -rf artifacts/behavior/data/raw/experiment-*\n")
        run.write(("#" if not self.config["workload_execute"] else "") + "doit behavior_execute_workloads\n")
        run.write(("#" if not self.config["workload_execute"] else "") + "mv artifacts/behavior/data/raw/experiment-* artifacts/behavior/data/raw/experiment\n")

        filters = "*"
        if "workload_filters" in self.config and len(self.config["workload_filters"]) > 0:
            filters = ",".join(self.config["workload_filters"])

        run.write(("#" if not self.config["workload_process"] else "") + f"doit behavior_perform_plan_extract_ou --glob_pattern={filters}\n")
        run.write(("#" if not self.config["workload_process"] else "") + "doit behavior_perform_plan_extract_qss\n")
        run.write(("#" if not self.config["workload_process"] else "") + "doit behavior_perform_plan_diff\n")
        run.write(("#" if not self.config["workload_process"] else "") + "doit behavior_perform_plan_state_merge\n")

    def output_train(self, run):
        run.write("\n")
        run.write("#############################\n")
        run.write("# Train Models\n")
        run.write("#############################\n")
        run.write(("#" if len(self.config["train_models"]) == 0 else "") + "rm -rf artifacts/behavior/models/*\n")
        for config in self.config["train_models"]:
            allowed_features = ",".join([
                "IndexOnlyScan_num_iterator_used",
                "IndexOnlyScan_num_defrag",
                "IndexScan_num_iterator_used",
                "IndexScan_num_outer_loops",
                "IndexScan_num_defrag",
                "ModifyTableInsert_num_extends",
                "ModifyTableIndexInsert_num_extends",
                "ModifyTableIndexInsert_num_splits",
                "ModifyTableUpdate_num_updates",
                "ModifyTableUpdate_num_extends",
                "ModifyTableDelete_num_deletes",
                "Agg_num_input_rows",
                "NestLoop_num_outer_rows",
                "NestLoop_num_inner_rows_cumulative"
            ])
            run.write(f"doit behavior_train --prefix_allow_derived_features=\"{allowed_features}\" ")
            run.write(f"--train_data=artifacts/behavior/data/merge/experiment/{config[0]}")
            if config[2]:
                run.write(f"--use_featurewiz=True")
            run.write("\n")
            run.write(f"mv artifacts/behavior/models/modeling_* artifacts/behavior/models/{config[1]}\n")
            run.write("\n")

    def output_eval_ou(self, run):
        run.write("\n")
        run.write("#############################\n")
        run.write("# Evaluate OU\n")
        run.write("#############################\n")
        run.write(("#" if len(self.config["evals_ou"]) == 0 else "") + "rm -rf artifacts/behavior/evals_ou/*\n")
        for config in self.config["evals_ou"]:
            run.write(f"doit behavior_eval_ou --eval_data=artifacts/behavior/data/merge/experiment/{config[0]} ")
            run.write(f"--models=artifacts/behavior/models/{config[1]}\n")
            run.write(f"mv artifacts/behavior/evals_ou/eval_* artifacts/behavior/evals_ou/{config[2]}\n")
            run.write("\n")

    def output_eval_query(self, run):
        run.write("\n")
        run.write("#############################\n")
        run.write("# Evaluate Query\n")
        run.write("#############################\n")
        run.write(("#" if len(self.config["eval_query"]) == 0 else "") + "rm -rf artifacts/behavior/evals_query/*\n")
        for config in self.config["eval_query"]:
            run.write("sudo pkill postgres || true\n")
            run.write("rm -rf /tmp/eval_query_scratch/\n")
            run.write(f"doit noisepage_init --config={os.getcwd()}/{config['pg_conf_path']}\n")
            run.write("doit benchbase_bootstrap_dbms\n")
            run.write(f"./artifacts/noisepage/pg_restore -j 8 -d benchbase {config['restore_db_path']}\n")
            run.write("./artifacts/noisepage/psql --dbname=benchbase --command=\"VACUUM\"\n")
            run.write("./artifacts/noisepage/psql --dbname=benchbase --command=\"CHECKPOINT\"\n")
            run.write("doit noisepage_qss_install --dbname=benchbase\n")
            run.write(f"doit benchbase_pg_analyze_benchmark --benchmark={config['benchmark']}\n")
            for i in range(len(config["model_subdir"])):
                run.write("doit behavior_eval_query \\\n")
                run.write(f"\t--psycopg2_conn=\"host=localhost port=5432 dbname=benchbase user={config['user']}\" \\\n")
                run.write(f"\t--num_iterations={config['num_iterations'][i]} \\\n")
                run.write(f"\t--predictive={config['predictive'][i]} \\\n")
                run.write(f"\t--session_sql={config['session_path'][i]} \\\n")
                run.write(f"\t--eval_raw_data=artifacts/behavior/data/raw/experiment/{config['raw_data'][i]} \\\n")
                run.write(f"\t--base_models=artifacts/behavior/models/{config['model_subdir'][i]}\n")
                run.write(f"mv artifacts/behavior/evals_query/eval_* artifacts/behavior/evals_query/{config['output_paths'][i]}\n")
                run.write("\n")
        run.write(("#" if not self.config["eval_query_plots"] else "") + "doit behavior_eval_query_plots\n")

    def output_workload_train(self, run):
        run.write("\n")
        run.write("#############################\n")
        run.write("# Train Workload Model\n")
        run.write("#############################\n")
        run.write(("#" if len(self.config["train_workload_models"]) == 0 else "") + "rm -rf artifacts/workload_analysis/workload_models/*\n")
        for config in self.config["train_workload_models"]:
            benchmarks = ",".join([c[1] for c in config["inputs"]])
            inputs = ",".join([f"artifacts/behavior/data/raw/experiment/{c[0]}" for c in config["inputs"]])
            input_train = ",".join([f"artifacts/behavior/data/raw/experiment/{c[0]}/train/*.feather" for c in config["inputs"]])
            epochs = config["epochs"]
            separate = config["separate"]
            batch_size = config["batch_size"]
            hidden_size = config["hidden_size"]
            output = config["output"]

            run.write(f"doit workload_analyze --input_workload={inputs} --output_workload={inputs} --benchmark={benchmarks} --workload_only=False\n")
            run.write(f"doit workload_populate_data --input_workload={inputs} --output_workload={inputs} --benchmark={benchmarks} --workload_only=False\n")
            run.write(f"doit workload_windowize --input_workload={inputs} --output_workload={inputs} --benchmark={benchmarks}\n")
            run.write(f"doit workload_prepare_train --input_workload={inputs} --output_workload={inputs} --benchmark={benchmarks}\n")
            run.write(f"doit workload_train --input_data={input_train} --output_dir={output} --epochs={epochs} --batch_size={batch_size} --hidden_size={hidden_size} --separate={separate}\n")
            run.write("\n")

    def output_eval_query_workload(self, run):
        run.write("\n")
        run.write("#############################\n")
        run.write("# Evaluate Query Workload Model\n")
        run.write("#############################\n")
        run.write(("#" if len(self.config["eval_query_workload"]) == 0 else "") + "rm -rf artifacts/behavior/evals_query_workload/*\n")
        for config in self.config["eval_query_workload"]:
            run.write("sudo pkill postgres || true\n")
            run.write("rm -rf /tmp/eval_query_workload_scratch/\n")
            run.write(f"doit noisepage_init --config={os.getcwd()}/{config['pg_conf_path']}\n")
            run.write("doit benchbase_bootstrap_dbms\n")
            run.write(f"./artifacts/noisepage/pg_restore -j 8 -d benchbase {config['restore_db_path']}\n")
            run.write("./artifacts/noisepage/psql --dbname=benchbase --command=\"VACUUM\"\n")
            run.write("./artifacts/noisepage/psql --dbname=benchbase --command=\"CHECKPOINT\"\n")
            run.write("doit noisepage_qss_install --dbname=benchbase\n")
            run.write(f"doit benchbase_pg_analyze_benchmark --benchmark={config['benchmark']}\n")

            conn = f"--psycopg2_conn=\"host=localhost port=5432 dbname=benchbase user={config['user']}\""
            inputs = f"artifacts/behavior/data/raw/experiment/{config['raw_data']}"
            benchmarks = config["benchmark"]
            run.write(f"doit workload_analyze --input_workload={inputs} --output_workload={inputs} --benchmark={benchmarks} --workload_only=True {conn}\n")
            run.write(f"doit workload_populate_data --input_workload={inputs} --output_workload={inputs} --benchmark={benchmarks} --workload_only=True {conn}\n")

            run.write("doit behavior_eval_query_workload \\\n")
            run.write(f"\t--benchmark={benchmarks} \\\n")
            run.write(f"\t--session_sql={config['session_path']} \\\n")
            run.write(f"\t--eval_raw_data={inputs} \\\n")
            run.write(f"\t--base_models=artifacts/behavior/models/{config['model_subdir']}\n")
            run.write(f"\t--workload_model=artifacts/workload_analysis/workload_models/{config['workload_model']}\n")
            run.write(f"\t{conn} \\\n")
            run.write(f"\t--slice_window={config['slice_window']} \\\n")

            run.write(f"mv artifacts/behavior/evals_query_workload/eval_* artifacts/behavior/evals_query_workload/{config['output_paths']}\n")
            if config["preserve_scratch"]:
                run.write(f"mv /tmp/eval_query_workload_scratch ~/npp/artifacts/behavior/evals_query_workload/{config['output_paths']}/\n")
            else:
                run.write(f"rm -rf /tmp/eval_query_workload_scratch\n")
            run.write("\n")
        run.write(("#" if not self.config["eval_query_workload_plots"] else "") + "doit behavior_eval_query_workload_plots\n")

    def main(self):
        config_path = Path(self.config_file)
        with config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)["skynet"]

        with open("run.sh", "w") as run:
            run.write("set -ex\n")
            run.write("sudo pkill postgres || true\n")

            self.output_generate(run)
            self.output_train(run)
            self.output_eval_ou(run)
            self.output_eval_query(run)
            self.output_workload_train(run)
            self.output_eval_query_workload(run)

            run.write("\n")
            run.write("#############################\n")
            run.write("# Output\n")
            run.write("#############################\n")
            run.write(f"mkdir -p {self.config['output_path']}\n")
            run.write("cd artifacts/behavior\n")
            run.write(("#" if not self.config["zip_data"] else "") + "tar zcf data.tgz data/merge data/raw\n")
            run.write(("#" if not self.config["zip_models"] else "") + "tar zcf models.tgz models\n")
            run.write(("#" if not self.config["zip_evals_ou"] else "") + "tar zcf evals_ou.tgz evals_ou\n")
            run.write(("#" if not self.config["zip_eval_query"] else "") + "tar zcf evals_query.tgz evals_query\n")
            run.write(("#" if not self.config["zip_eval_query_workload"] else "") + "tar zcf evals_query_workload.tgz evals_query_workload\n")
            run.write(("#" if not self.config["zip_eval_query_plots"] else "") + "tar zcf evals_query_plots.tgz evals_query/*/*/plots\n")
            run.write(("#" if not self.config["zip_eval_query_workload_plots"] else "") + "tar zcf evals_query_workload_plots.tgz evals_query_workload/*/plots\n")
            run.write(("#" if not self.config["zip_data"] else "") + f"mv data.tgz {self.config['output_path']}\n")
            run.write(("#" if not self.config["zip_models"] else "") + f"mv models.tgz {self.config['output_path']}\n")
            run.write(("#" if not self.config["zip_evals_ou"] else "") + f"mv evals_ou.tgz {self.config['output_path']}\n")
            run.write(("#" if not self.config["zip_eval_query"] else "") + f"mv evals_query.tgz {self.config['output_path']}\n")
            run.write(("#" if not self.config["zip_eval_query_workload"] else "") + f"mv evals_query_workload.tgz {self.config['output_path']}\n")
            run.write(("#" if not self.config["zip_eval_query_plots"] else "") + f"mv evals_query_plots.tgz {self.config['output_path']}\n")
            run.write(("#" if not self.config["zip_eval_query_workload_plots"] else "") + f"mv evals_query_workload_plots.tgz {self.config['output_path']}\n")
            run.write("cd artifacts/workload_analysis\n")
            run.write(("#" if not self.config["zip_workload_models"] else "") + "tar zcf workload_models.tgz workload_models\n")
            run.write(("#" if not self.config["zip_workload_models"] else "") + f"mv workload_models.tgz {self.config['output_path']}\n")
            run.write("cd ../../\n")


if __name__ == "__main__":
    SkynetCLI.run()
