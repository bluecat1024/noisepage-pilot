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
from behavior.model_query import eval_query_plots

logger = logging.getLogger(__name__)


class BehaviorCLI(cli.Application):
    def main(self) -> None:
        pass


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s:%(asctime)s %(message)s", level=logging.INFO)
    BehaviorCLI.subcommand("skynet", skynet.SkynetCLI)
    BehaviorCLI.subcommand("generate_workloads", generate_workloads.GenerateWorkloadsCLI)
    BehaviorCLI.subcommand("extract_ou", extract_ou.ExtractOUCLI)
    BehaviorCLI.subcommand("extract_qss", extract_qss.ExtractQSSCLI)
    BehaviorCLI.subcommand("diff", diff.DiffCLI)
    BehaviorCLI.subcommand("state_merge", state_merge.StateMergeCLI)
    BehaviorCLI.subcommand("train", train.TrainCLI)
    BehaviorCLI.subcommand("eval_ou", eval_ou.EvalOUCLI)
    BehaviorCLI.subcommand("eval_query", eval_query.EvalQueryCLI)
    BehaviorCLI.subcommand("eval_query_plots", eval_query_plots.EvalQueryPlotsCLI)
    BehaviorCLI.subcommand("collector", pg_collector.CollectorCLI)
    BehaviorCLI.run()
