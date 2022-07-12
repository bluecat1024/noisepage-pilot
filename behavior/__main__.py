import logging

from plumbum import cli

from behavior import skynet
from behavior.datagen import generate_workloads
from behavior.plans import extract_ou
from behavior.plans import extract_qss
from behavior.plans import diff
from behavior.plans import state_merge
from behavior.modeling import train
from behavior.modeling import eval_ou
from behavior.modeling import eval_query
from behavior.modeling import eval_query_plots

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
    BehaviorCLI.run()
