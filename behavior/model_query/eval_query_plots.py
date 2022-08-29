import glob
import logging
import shutil
import copy
from datetime import datetime
import numpy as np
import itertools
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from plumbum import cli

logger = logging.getLogger(__name__)

def generate_holistic(query_stream, dir_output):
    def plot(df, name):
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 7.2))

        df.predicted_minus_elapsed.plot.kde(color='r', ax=ax)
        percentiles = [2.5, 5, 25, 50, 75, 95, 97.5]
        percents = np.percentile(df.predicted_minus_elapsed, percentiles)
        bounds = [df.predicted_minus_elapsed.min(), df.predicted_minus_elapsed.max()]
        ax.scatter(bounds, [0, 0], color='b')
        ax.scatter(percents, np.zeros(len(percentiles)), color='g')
        ax.set_xlabel("predicted - elapsed")

        plt.savefig(dir_output / f"{name}.png")
        plt.close()

    plot(query_stream, "all_queries")
    plot(query_stream[query_stream.num_modify == 0], "all_select")
    plot(query_stream[query_stream.num_modify != 0], "all_modify")


def generate_per_query_plots(query_stream, dir_output):
    # Output error distribution plots based on query_id.
    qid_groups = query_stream.groupby(by=["query_id"])
    for group in qid_groups:
        logger.info("Processing query %s", group[0])
        fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2))
        ax = axes[0]

        # Plot elapsed and predicted elapsed time on the same graph as a scatter.
        group[1].plot(title=f"qid: {group[0]}", x="order", y="elapsed_us", color='r', ax=ax, kind='scatter')
        group[1].plot(title=f"qid: {group[0]}", x="order", y="pred_elapsed_us", color='b', ax=ax, kind='scatter')
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

        plt.savefig(dir_output / f"{group[0]}.png")
        plt.close()


def generate_predicted_query_error(query_stream, dir_output, min_error_threshold, max_error_threshold):
    x = 0
    width = 0.8
    ticks = []
    labels = []
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2))

    txtout = open(dir_output / "select_error.txt", "w")
    frame = query_stream[(query_stream.num_modify == 0)]
    for group in frame.groupby(by=["modify_target"]):
        ticks.append(x)
        labels.append(group[0])

        qgroups = group[1].groupby(by=["query_id"]).sum()
        colors = plt.cm.BuPu(np.linspace(0.25, 1.0, len(qgroups)))

        idx = 0
        for qgroup in qgroups.itertuples():
            error = qgroup.abs_diff / qgroup.cnt
            if error > min_error_threshold and (max_error_threshold is None or error < max_error_threshold):
                rect = axes[0].bar(x, qgroup.abs_diff / qgroup.cnt, width=width, color=colors[idx])[0]
                axes[0].text(rect.get_x() + rect.get_width() / 2., 1.05 * rect.get_height(), '%d' % error, ha='center', va='bottom')

                rect = axes[1].bar(x, qgroup.elapsed_us / qgroup.cnt, width=width, color=colors[idx])[0]
                axes[1].text(rect.get_x() + rect.get_width() / 2., 1.05 * rect.get_height(), '%d' % (qgroup.elapsed_us / qgroup.cnt), ha='center', va='bottom')
                txtout.write(f"{qgroup.Index} avg. predicted/true: {error} / {qgroup.elapsed_us / qgroup.cnt}\n")

                x += width
                idx += 1

        x += 1

    axes[0].set_ylabel("Avg. Absolute Error (sum(|pred-elapsed|) / cnt)")
    axes[0].set_xticks(ticks, minor=False)
    axes[0].set_xticklabels(labels, fontdict=None, minor=False)

    axes[1].set_ylabel("Avg. Elapsed Runtime (sum(elapsed) / cnt)")
    axes[1].set_xticks(ticks, minor=False)
    axes[1].set_xticklabels(labels, fontdict=None, minor=False)
    plt.savefig(dir_output / "summary_select.png")
    plt.close()
    txtout.close()


def generate_predicted_query_error_modify(query_stream, dir_output):
    x = 0
    width = 0.8
    ticks = []
    labels = []
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2))
    txtout = open(dir_output / "modify_error.txt", "w")

    frame = query_stream[(query_stream.num_modify != 0)]

    for group in frame.groupby(by=["modify_target"]):
        ticks.append(x)
        labels.append(group[0])
        colors = plt.cm.BuPu(np.linspace(0.25, 1.0, 3))
        idx = 0
        for t, subframe in enumerate([group[1][group[1].is_insert != 0], group[1][group[1].is_update != 0], group[1][group[1].is_delete != 0]]):
            if t == 0:
                txtout.write(f"{group[0]} INSERT\n")
            elif t == 1:
                txtout.write(f"{group[0]} UPDATE\n")
            else:
                txtout.write(f"{group[0]} DELETE\n")

            qgroups = subframe.groupby(by=["query_id"]).sum()
            for qgroup in qgroups.itertuples():
                error = qgroup.abs_diff / qgroup.cnt
                rect = axes[0].bar(x, qgroup.abs_diff / qgroup.cnt, width=width, color=colors[idx])[0]
                axes[0].text(rect.get_x() + rect.get_width() / 2., 1.05 * rect.get_height(), '%d' % error, ha='center', va='bottom')

                rect = axes[1].bar(x, qgroup.elapsed_us / qgroup.cnt, width=width, color=colors[idx])[0]
                axes[1].text(rect.get_x() + rect.get_width() / 2., 1.05 * rect.get_height(), '%d' % (qgroup.elapsed_us / qgroup.cnt), ha='center', va='bottom')
                txtout.write(f"{qgroup.Index} avg. predicted/true: {error} / {qgroup.elapsed_us / qgroup.cnt}\n")
                x += width

            idx += 1

        txtout.write("\n")
        x += 1

    axes[0].set_ylabel("Avg. Absolute Error (sum(|pred-elapsed|) / cnt)")
    axes[0].set_xticks(ticks, minor=False)
    axes[0].set_xticklabels(labels, fontdict=None, minor=False)

    axes[1].set_ylabel("Avg. Elapsed Runtime (sum(elapsed) / cnt)")
    axes[1].set_xticks(ticks, minor=False)
    axes[1].set_xticklabels(labels, fontdict=None, minor=False)
    plt.savefig(dir_output / "summary_modification.png")
    plt.close()
    txtout.close()


def generate_plots(query_stream, dir_output):
    with open(dir_output / "summary.txt", "w") as f:
        f.write(f"Total Elapsed Us: {query_stream.elapsed_us.sum()}\n")
        f.write(f"Total Predicted Elapsed Us: {query_stream.pred_elapsed_us.sum()}\n")

        def summary(df, prefix):
            f.write(f"Average Absolute Error ({prefix}): {(df.abs_diff.sum() / df.shape[0])}\n")

            box = [1, 5, 10, 100]
            for bound in box:
                f.write(f"% within absolute error {bound} ({prefix}): {(len(df[df.abs_diff < bound]) / df.shape[0])}\n")
            f.write(f"% exceed absolute error {box[-1]} ({prefix}): {(len(df[df.abs_diff >= box[-1]]) / df.shape[0])}\n")

            for bound in box:
                f.write(f"% underpredict error {bound} ({prefix}): {(len(df[(df.predicted_minus_elapsed < 0) & (df.abs_diff < bound)]) / df.shape[0])}\n")
            f.write(f"% exceed underpredict error {box[-1]} ({prefix}): {(len(df[(df.predicted_minus_elapsed < 0) & (df.abs_diff >= box[-1])]) / df.shape[0])}\n")

        select = query_stream[query_stream.num_modify == 0]
        modify = query_stream[query_stream.num_modify != 0]
        summary(query_stream, "All Queries")
        f.write("\n")

        summary(select, "SELECT")
        f.write("\n")

        summary(modify, "MODIFY")

    generate_holistic(query_stream, dir_output)
    generate_per_query_plots(query_stream, dir_output)
    generate_predicted_query_error(query_stream, dir_output, 0, None)
    generate_predicted_query_error_modify(query_stream, dir_output)



def main(dir_input):
    inputs = dir_input.rglob("query_results.feather")
    for input_result in inputs:
        logger.info("Processing: %s", input_result)
        dir_output = input_result.parent / "plots"
        dir_output.mkdir(parents=True, exist_ok=True)

        query_stream = pd.read_feather(input_result)
        query_stream["predicted_minus_elapsed"] = query_stream["pred_elapsed_us"] - query_stream["elapsed_us"]
        query_stream["abs_diff"] = query_stream.predicted_minus_elapsed.apply(lambda c: abs(c))
        query_stream["cnt"] = 1

        consider_streams = [("original", query_stream),]
        if Path(input_result.parent / "ddl_changes.pickle").exists():
            with open(f"{input_result.parent}/ddl_changes.pickle", "rb") as f:
                ff_tbl_change_map = pickle.load(f)

            if len(ff_tbl_change_map) > 0:
                # We just use an arbitrary table's timestamps to segment.
                # The reason we do this: no queries run during a period of DDL changes.
                slots = ff_tbl_change_map[list(ff_tbl_change_map.keys())[0]]
                for i, (ts, _) in enumerate(slots):
                    consider_streams.append((f"{i}", query_stream[query_stream.statement_timestamp < ts]))
                    query_stream = query_stream[query_stream.statement_timestamp >= ts]

                if query_stream.shape[0] > 0:
                    consider_streams.append((f"{len(slots)}", query_stream))

        for (flag, query_stream) in consider_streams:
            if flag == "original":
                generate_plots(query_stream, dir_output)
            else:
                sub_output = dir_output / f"{flag}"
                sub_output.mkdir(parents=True, exist_ok=True)
                generate_plots(query_stream, sub_output)


class EvalQueryPlotsCLI(cli.Application):
    dir_input = cli.SwitchAttr(
        "--dir-input",
        Path,
        mandatory=True,
        help="Path to the input folder to recursively search.",
    )

    def main(self):
        main(self.dir_input)


if __name__ == "__main__":
    EvalQueryPlotsCLI.run()

