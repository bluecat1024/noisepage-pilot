from datetime import datetime
import numpy as np
import itertools
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from plumbum import cli
from behavior import Targets
from behavior.modeling.utils import evaluate_ou_model


def create_plots(output_dir, method, raw_df, preds_path):
    raw_df["predicted_minus_elapsed"] = raw_df[f"pred_{Targets.ELAPSED_US.value}"] - raw_df[Targets.ELAPSED_US.value]
    qid_groups = raw_df.groupby(by=["query_id"])

    for group in qid_groups:
        fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2))
        ax = axes[0]

        # Plot elapsed and predicted elapsed time on the same graph as a scatter.
        group[1].plot(title=f"qid: {group[0]}", x="unix_timestamp", y=Targets.ELAPSED_US.value, color='r', ax=ax, kind='scatter')
        group[1].plot(title=f"qid: {group[0]}", x="unix_timestamp", y=f"pred_{Targets.ELAPSED_US.value}", color='b', ax=ax, kind='scatter')
        ax.set_xticks([])

        if len(group[1]) > 1 and len(group[1].predicted_minus_elapsed.value_counts()) > 1:
            # Only plot the second graph if there is more than 1 distinct value.
            ax = axes[1]
            group[1].predicted_minus_elapsed.plot.kde(color='r', ax=ax)

            percentiles = [2.5, 5, 25, 50, 75, 95, 97.5]
            percents = np.percentile(group[1].predicted_minus_elapsed, percentiles)
            bounds = [group[1].predicted_minus_elapsed.min(), group[1].predicted_minus_elapsed.max()]
            ax.scatter(bounds, [0, 0], color='b')
            ax.scatter(percents, np.zeros(len(percentiles)), color='g')
            ax.set_xlabel("predicted - elapsed")

        plt.suptitle(preds_path)

        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{group[0]}_{preds_path.stem}_{method}.png")
        plt.close()


def main(dir_models, methods, dir_data, dir_evals_output, generate_plots):
    eval_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output = dir_evals_output / f"eval_{eval_timestamp}"

    output_evals = base_output / "evals"
    output_evals.mkdir(parents=True, exist_ok=True)
    output_plots_path = None
    if generate_plots:
        output_plots_path = base_output / f"plots"
        output_plots_path.mkdir(parents=True, exist_ok=True)

    # Get all the models in the model path.
    for model_path in dir_models.rglob('*.pkl'):
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)

        if methods is not None and len(methods) > 0:
            # In this case, the method is not requested so ignore it.
            if model.method not in methods:
                continue

        # Evaluation is expected of form [experiment]/[benchmark]/[OU].feather
        for feather in dir_data.rglob('*.feather'):
            if feather.name.startswith(model.ou_name):
                print("Evaluating ", model.method, model.ou_name, "on data", feather)

                # Tack the OU onto the output directory.
                # This produces evals/eval_[timestamp]/[OU]/[method]_[benchmark]_[feather.stem]_preds/stats files."
                output_dir = output_evals / model.ou_name
                output_dir.mkdir(parents=True, exist_ok=True)
                preds_df = evaluate_ou_model(model, output_dir, feather.parts[-2], eval_file=feather, return_df=generate_plots)

                if generate_plots:
                    # Generate the plots if needed.
                    # Format plots/plots_[timestamp]/[benchmark]/[qid]_[feather.stem].png"
                    output_plot_dir = output_plots_path / feather.parts[-2]
                    create_plots(output_plot_dir, model.method, preds_df, feather)


class EvalOUCLI(cli.Application):
    dir_data = cli.SwitchAttr(
        "--dir-data",
        Path,
        mandatory=True,
        help="Folder containing evaluation CSVs. (structure: [experiment]/[benchmark]/*.feather)",
    )
    dir_evals_output = cli.SwitchAttr(
        "--dir-evals-output",
        Path,
        mandatory=True,
        help="Folder to output evaluations to.",
    )
    generate_plots = cli.Flag(
        "--generate-plots",
        default = False
    )
    dir_models = cli.SwitchAttr(
        "--dir-models",
        Path,
        mandatory=True,
        help="Folder to extract models from (generated by doit behavior_train).",
    )
    methods = cli.SwitchAttr(
        "--methods",
        mandatory=False,
        help="Comma separated model methods that should be evaluated.",
    )

    def main(self):
        methods = [] if not self.methods else self.methods.split(",")
        main(self.dir_models, methods, self.dir_data, self.dir_evals_output, self.generate_plots)


if __name__ == "__main__":
    EvalOUCLI.run()
