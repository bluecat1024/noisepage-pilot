import argparse
import gc
import glob
import yaml
import logging
import shutil
import copy
from tqdm import tqdm
from datetime import datetime
import numpy as np
import itertools
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from plumbum import cli

logger = logging.getLogger(__name__)

def generate_per_query_plots(query_stream, num_models, dir_output):
    # Output error distribution plots based on query_id.
    qid_groups = query_stream.groupby(by=["query_id"])
    (Path(dir_output) / "hists").mkdir(parents=True, exist_ok=True)
    (Path(dir_output) / "qerrors").mkdir(parents=True, exist_ok=True)
    (Path(dir_output) / "scatter").mkdir(parents=True, exist_ok=True)
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    for group in qid_groups:
        print("Processing query ", group[0])
        fig, axes = plt.subplots(1, 1, figsize=(25.6, 14.4))

        # Make the histogram.
        legend = []
        for i, sbgrp in enumerate(group[1].groupby(by=["model"])):
            sbgrp[1].pred_elapsed_us.hist(bins=1000, color=colors[i], ax=axes, alpha=0.5)
            legend.append(sbgrp[0])
        axes.legend(legend)
        plt.savefig(dir_output / "hists" / f"{group[0]}.png")
        plt.close()

        # Make the qerror histogram.
        legend = []
        fig, axes = plt.subplots(1, 1, figsize=(25.6, 14.4))
        for i, sbgrp in enumerate(group[1].groupby(by=["model"])):
            sbgrp[1].qerror.hist(bins=1000, color=colors[i], ax=axes, alpha=0.5)
            legend.append(sbgrp[0])
        axes.legend(legend)
        plt.savefig(dir_output / "qerrors" / f"{group[0]}.png")
        plt.close()

        # Plot the scatter within bounds.
        legend = []
        fig, axes = plt.subplots(num_models, 1, figsize=(25.6, 14.4))
        upper = group[1].elapsed_us.max()
        mean = group[1].elapsed_us.mean()
        for i, sbgrp in enumerate(group[1].groupby(by=["model"])):
            sbgrp[1].plot.scatter(title=f"qid: {group[0]} {sbgrp[0]} model", x="query_order", y="elapsed_us", c='r', ax=axes[i])
            sbgrp[1].plot.scatter(title=f"qid: {group[0]} {sbgrp[0]} model", x="query_order", y="pred_elapsed_us", c='b', ax=axes[i])
            axes[i].legend(['Elapsed', 'Predicted'])
            if upper > mean + 500:
                axes[i].set_ylim(0, mean+500)
            axes[i].set_xticks([])

        plt.savefig(dir_output / "scatter" / f"{group[0]}.png")
        plt.close()


def generate_predicted_query_error(query_stream, select, input_models, dir_output):
    width = 0.8
    fig, axes = plt.subplots(1, 1, figsize=(25.6, 14.4))

    y_lim = 300
    axes.set_ylim(0, y_lim)

    # Filter the frame out depending on whether select or not.
    if select:
        frame = query_stream[(query_stream.OP == "SELECT")]
    else:
        frame = query_stream[(query_stream.OP != "SELECT")]

    x_labels = []
    labels = []

    x = 0
    legend = input_models
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    hatches = [None, '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    for group in frame.groupby(by=["target"]):
        if select:
            it = enumerate([group[1]])
        else:
            it = enumerate([group[1][group[1].OP == "INSERT"], group[1][group[1].OP == "UPDATE"], group[1][group[1].OP == "DELETE"]])

        for t, subframe in it:
            lbl = ["SELECT"] if select else ["INSERT", "UPDATE", "DELETE"]
            if subframe.shape[0] == 0:
                continue

            labels.append(group[0] + " " + lbl[t])
            x_labels.append(x)

            for sgrp in subframe.groupby(by=["query_id"]):
                for i, m in enumerate(input_models):
                    mo = sgrp[1][sgrp[1].model == m]
                    mo_rect = axes.bar(x, mo.abs_diff.sum() / len(mo), width, color=colors[i], hatch=hatches[i], alpha=0.5)
                    x += width
                x += 0.2
            x += 1.6

    axes.set_ylabel("Avg. Absolute Error (us)")
    axes.set_xticks(x_labels, labels)
    axes.legend(legend)

    for c in axes.containers:
        assert len(c.datavalues) == 1
        if c.datavalues[0] > y_lim:
            for r in c:
                axes.text(r.get_x(), 300, "{s}".format(s=int(c.datavalues[0])), clip_on=False)
        else:
            axes.bar_label(c, padding=3, fmt="%d", clip_on=False)

    if select:
        axes.set_title(f"SELECT Error")
        plt.savefig(dir_output / "summary_select.png")
    else:
        axes.set_title(f"MODIFY Error")
        plt.savefig(dir_output / "summary_modify.png")

    plt.close()


def generate_txn_plots(txn_analysis_file, query_stream, models, output_dir):
    query_stream.sort_values(by=["statement_timestamp"], inplace=True)
    with open(txn_analysis_file, "r") as f:
        analysis = yaml.load(f, Loader=yaml.FullLoader)

    if not Path(f"/tmp/analysis_cmp/done").exists():
        query_stream["query_text"] = query_stream.query_text.str.lower()
        txn_groupings = {txn_name: [] for txn_name in analysis}
        for (model, txn), txn_queries in tqdm(query_stream.groupby(by=["model", "txn"])):
            valid_txn_name = None
            for txn_name, id_queries_set in analysis.items():
                for id_queries in id_queries_set:
                    valid = True
                    for i, query_text in enumerate(id_queries):
                        if i < txn_queries.shape[0] and query_text in txn_queries.iloc[i].query_text:
                            continue

                        valid = False
                        break

                    if valid:
                        valid_txn_name = txn_name
                        break

                if valid_txn_name is not None:
                    break

            assert valid_txn_name is not None, print(txn_queries.query_text)
            txn_groupings[valid_txn_name].append({"model": model, "txn_elapsed_us": txn_queries.elapsed_us.sum(), "txn_pred_elapsed_us": txn_queries.pred_elapsed_us.sum()})

        txn_groupings = {txn: pd.DataFrame(v) for txn, v in txn_groupings.items()}

        Path(f"/tmp/analysis_cmp/").mkdir(parents=True, exist_ok=True)
        for txn_group, v in txn_groupings.items():
            v.to_feather(f"/tmp/analysis_cmp/{txn_group}.feather")
        open(f"/tmp/analysis_cmp/done", "w").close()
    else:
        files = glob.glob(f"/tmp/analysis_cmp/*.feather")
        txn_groupings = {}
        for f in files:
            name = Path(f).parts[-1].split(".feather")[0]
            txn_groupings[name] = pd.read_feather(f"/tmp/analysis_cmp/{name}.feather")

    for txn_name, points in txn_groupings.items():
        print(f"Processing the {txn_name} txn grouping")
        df = pd.DataFrame(points)

        # Create the true/elapsed KDE for each of the models.
        fig, axes = plt.subplots(len(models), 1, figsize=(25.6, 14.4))
        x_min = min(df.txn_elapsed_us.min(), df.txn_pred_elapsed_us.min())
        x_max = max(df.txn_elapsed_us.max(), df.txn_pred_elapsed_us.max())
        for i, m in enumerate(models):
            df_m = df[df.model == m]
            sns.kdeplot(df_m.txn_elapsed_us, color='r', ax=axes[i], label='True Dist')
            sns.kdeplot(df_m.txn_pred_elapsed_us, color='b', ax=axes[i], alpha=0.5, fill=True, label='Predicted Dist')
            axes[i].set_title(f"{txn_name} {m} Model Elapsed Us")
            axes[i].legend(["True Dist", "Predicted Dist"])
        plt.savefig(output_dir / f"{txn_name}_kde.png")
        plt.close()

        # Create the true/elapsed histogram for each of the models.
        fig, axes = plt.subplots(len(models), 1, figsize=(25.6, 14.4))
        for i, m in enumerate(models):
            df_m = df[df.model == m]
            axes[i].set_xlim(x_min, x_max)
            df_m.txn_elapsed_us.hist(bins=1000, color='r', ax=axes[i], alpha=1.0, label="True Dist")
            df_m.txn_pred_elapsed_us.hist(bins=1000, color='b', ax=axes[i], alpha=0.5, hatch='/', label='Predicted Dist')
            axes[i].set_title(f"{txn_name} {m} Model Elapsed Us (Histogram)")
            axes[i].legend(["True Dist", "Predicted Dist"])
        plt.savefig(output_dir / f"{txn_name}_hist.png")
        plt.close()

        # Create the true/elapsed CDF for each of the models.
        fig, axes = plt.subplots(len(models), 1, figsize=(25.6, 14.4))
        for i, m in enumerate(models):
            df_m = df[df.model == m]
            df_m.txn_elapsed_us.hist(cumulative=True, density=True, histtype='step', bins=1000, ax=axes[i], color='r', alpha=0.5, label="True Dist")
            df_m.txn_pred_elapsed_us.hist(cumulative=True, density=True, histtype='step', bins=1000, ax=axes[i], color='b', alpha=0.5, label="Predicted Dist")
            axes[i].set_title(f"{txn_name} {m} Model Elapsed Us (CDF)")
            axes[i].legend(["True Dist", "Predicted Dist"])
            axes[i].set_xlim(x_min, x_max)
        plt.savefig(output_dir / f"{txn_name}_cdf.png")
        plt.close()

    shutil.rmtree(f"/tmp/analysis_cmp")


def main(dir_input, input_names, dir_output, txn_analysis_file, generate_per_query, generate_predict_abs_errors):
    inputs = [(a, b) for a, b in zip(dir_input.split(","), input_names.split(","))]
    input_streams = []
    for (input_result, t) in inputs:
        query_stream = pd.read_feather(input_result)
        query_stream["predicted_minus_elapsed"] = query_stream["pred_elapsed_us"] - query_stream["elapsed_us"]
        query_stream["abs_diff"] = query_stream.predicted_minus_elapsed.apply(lambda c: abs(c))
        query_stream["qerror"] = np.maximum(query_stream.elapsed_us / query_stream.pred_elapsed_us, query_stream.pred_elapsed_us / query_stream.elapsed_us)
        query_stream["cnt"] = 1
        query_stream["model"] = t

        if "query_order" not in query_stream:
            query_stream.sort_values(by=["statement_timestamp"], ignore_index=True, inplace=True)
            query_stream["query_order"] = query_stream.index
        input_streams.append(query_stream)

    df = pd.concat(input_streams, ignore_index=True)
    del input_streams

    if generate_per_query:
        generate_per_query_plots(df, input_names.split(","), dir_output)

    if generate_predict_abs_errors:
        generate_predicted_query_error(df, True, input_names.split(","), dir_output)
        generate_predicted_query_error(df, False, input_names.split(","), dir_output)

    if txn_analysis_file:
        generate_txn_plots(txn_analysis_file, df, input_names.split(","), dir_output)


class EvalQueryComparePlotsCLI(cli.Application):
    dir_input = cli.SwitchAttr(
        "--dir-input",
        str,
        mandatory=True,
        help="List of input paths to use for comparative plotting.",
    )

    input_names = cli.SwitchAttr(
        "--input-names",
        str,
        mandatory=True,
        help="List of input model names.",
    )

    dir_output = cli.SwitchAttr(
        "--dir-output",
        Path,
        mandatory=True,
        help="Path to the output of where to store plots.",
    )

    txn_analysis_file = cli.SwitchAttr(
        "--txn-analysis-file",
        Path,
        help="Path to transaction analysis file.",
    )

    generate_per_query = cli.Flag(
        "--generate-per-query",
        default=False,
        help="Whether to generate per-query plots of the errors.",
    )

    generate_predict_abs_errors = cli.Flag(
        "--generate-predict-abs-errors",
        default=False,
        help="Whether to generate abs errors against each table.",
    )

    def main(self):
        main(self.dir_input, self.input_names, self.dir_output, self.txn_analysis_file, self.generate_per_query, self.generate_predict_abs_errors)


if __name__ == "__main__":
    EvalQueryComparePlotsCLI.run()

