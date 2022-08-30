import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader

import logging
from plumbum import cli
from behavior.model_workload.model import WorkloadModel, MODEL_WORKLOAD_TARGETS


logger = logging.getLogger("eval")


def get_dataset(glob_pattern):
    patterns = glob_pattern.split(",")
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))

    data = pd.concat(map(pd.read_feather, all_files))
    data.reset_index(drop=True, inplace=True)
    return data


def loss_fn(preds, labels):
    loss = MSELoss()
    return loss(preds, labels)


def print_loss(preds, labels):
    loss_segs = []
    for j in range(len(preds[0])):
        loss_seg = []
        for i in range(len(preds)):
            loss = abs(preds[i][j].numpy() - labels[i][j])
            loss_seg.append(loss)
        loss_segs.append(loss_seg)

    for i in range(len(loss_segs)):
        logger.info("L1 Statistics for target: %s", MODEL_WORKLOAD_TARGETS[i])
        loss_seg = loss_segs[i]

        logger.info("Average: %s (%s)", np.mean(loss_seg), np.mean([pred[i].item() for pred in preds]))
        logger.info("Median: %s", np.median(loss_seg))
        logger.info("90th percentile: %s", np.percentile(loss_seg, 90))
        logger.info("95th percentile: %s", np.percentile(loss_seg, 95))
        logger.info("99th percentile: %s", np.percentile(loss_seg, 99))
        logger.info("Max: %s", np.max(loss_seg))
        logger.info("Mean: %s", np.mean(loss_seg))
        logger.info("")


def evaluate(patterns, model_input, batch_size):
    model = WorkloadModel()
    model.load(model_input)
    with torch.no_grad():
        test_df = get_dataset(patterns)
        test_data = model.prepare_inputs(test_df, train=False)
        test_data_loader = DataLoader(test_data, batch_size=batch_size)

        preds = []
        targets = []
        for batch_idx, data_batch in enumerate(test_data_loader):
            outputs = model.predict(*data_batch)
            preds.extend(outputs)

        outputs = [[p.item() for p in pred] for pred in preds]
        preds_df = pd.DataFrame(outputs, columns=["pred_" + t for t in MODEL_WORKLOAD_TARGETS])
        pd.concat([test_df, preds_df], axis=1).to_feather("/tmp/preds.feather")

        for tup in test_df[MODEL_WORKLOAD_TARGETS].itertuples(index=False):
            targets.append([val for val in tup])

        print_loss(preds, targets)


class WorkloadEvaluateCLI(cli.Application):
    dir_input = cli.SwitchAttr(
        "--dir-input",
        str,
        mandatory=True,
        help="Path to the folder containing the input data.",
    )

    model_input = cli.SwitchAttr(
        "--model-input",
        str,
        mandatory=True,
        help="Path to the folder containing the model.",
    )

    batch_size = cli.SwitchAttr(
        "--batch-size",
        int,
        help="Size of the batch that should be used.",
    )

    def main(self):
        evaluate(self.dir_input, self.model_input, self.batch_size)


if __name__ == "__main__":
    WorkloadEvaluateCLI.run()
