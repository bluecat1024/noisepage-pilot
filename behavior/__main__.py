import logging

from plumbum import cli

from behavior import skynet
from behavior.datagen import generate_workloads
from behavior.datagen import pg_collector
from behavior.model_ous.process import extract_ou
from behavior.model_ous.process import extract_qss
from behavior.model_ous.process import diff
from behavior.model_ous.process import state_merge
from behavior.model_ous import train
from behavior.model_ous import eval_ou
from behavior.model_query import eval_query
from behavior.model_query import plot

from behavior.model_workload import analyze
from behavior.model_workload import populate_data
from behavior.model_workload import windowize
from behavior.model_workload import generate_train
from behavior.model_workload import train as workload_train
from behavior.model_query import eval_query_workload

logger = logging.getLogger(__name__)


class BehaviorCLI(cli.Application):
    def main(self) -> None:
        pass


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s:%(asctime)s %(message)s", level=logging.INFO)
    BehaviorCLI.subcommand("skynet", skynet.SkynetCLI)
    BehaviorCLI.subcommand("generate_workloads", generate_workloads.GenerateWorkloadsCLI)
    BehaviorCLI.subcommand("collector", pg_collector.CollectorCLI)

    BehaviorCLI.subcommand("extract_ou", extract_ou.ExtractOUCLI)
    BehaviorCLI.subcommand("extract_qss", extract_qss.ExtractQSSCLI)
    BehaviorCLI.subcommand("diff", diff.DiffCLI)
    BehaviorCLI.subcommand("state_merge", state_merge.StateMergeCLI)
    BehaviorCLI.subcommand("train", train.TrainCLI)
    BehaviorCLI.subcommand("eval_ou", eval_ou.EvalOUCLI)

    BehaviorCLI.subcommand("workload_analyze", analyze.AnalyzeWorkloadCLI)
    BehaviorCLI.subcommand("workload_populate_data", populate_data.PopulateDataCLI)
    BehaviorCLI.subcommand("workload_windowize", windowize.WindowizeCLI)
    BehaviorCLI.subcommand("workload_prepare_train", generate_train.GenerateTrainCLI)
    BehaviorCLI.subcommand("workload_train", workload_train.WorkloadTrainCLI)
    BehaviorCLI.subcommand("eval_query_workload", eval_query_workload.EvalQueryWorkloadCLI)

    BehaviorCLI.subcommand("eval_query", eval_query.EvalQueryCLI)
    BehaviorCLI.subcommand("eval_query_plots", plot.EvalQueryPlotsCLI)

    BehaviorCLI.run()
