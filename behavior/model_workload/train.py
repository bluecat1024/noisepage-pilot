import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.nn import MSELoss
from torch.utils.data import dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader

import logging
from plumbum import cli
from behavior.model_workload.model import WorkloadModel, MODEL_WORKLOAD_TARGETS


logger = logging.getLogger("train")


def get_dataset(glob_pattern, sample):
    patterns = glob_pattern.split(",")
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))

    data = pd.concat(map(pd.read_feather, all_files))
    data.reset_index(drop=True, inplace=True)
    if sample > 0:
        val = data.sample(frac=sample)
        data.drop(val.index, inplace=True)
    else:
        val = None

    return data, val


def loss_fn(preds, labels):
    loss = MSELoss()
    return loss(preds, labels)


def print_loss(preds, labels):
    loss_segs = []
    assert len(preds) > 0
    for j in range(len(preds[0])):
        loss_seg = []
        for i in range(len(preds)):
            loss = abs(preds[i][j].numpy() - labels[i][j].numpy())
            loss_seg.append(loss)
        loss_segs.append(loss_seg)

    for i in range(len(loss_segs)):
        logger.info("L1 Statistics for target: %s", MODEL_WORKLOAD_TARGETS[i])
        loss_seg = loss_segs[i]

        logger.info("Average: %s", np.mean(loss_seg))
        logger.info("Median: %s", np.median(loss_seg))
        logger.info("90th percentile: %s", np.percentile(loss_seg, 90))
        logger.info("95th percentile: %s", np.percentile(loss_seg, 95))
        logger.info("99th percentile: %s", np.percentile(loss_seg, 99))
        logger.info("Max: %s", np.max(loss_seg))
        logger.info("Mean: %s", np.mean(loss_seg))
        logger.info("")


def predict(model, data_loader, separate):
    preds = []
    targets = []

    for batch_idx, data_batch in enumerate(data_loader):
        outputs = model.predict(*(data_batch[0:-1]))
        target = [t for t in data_batch[-1]]

        preds.extend(outputs)
        targets.extend(target)

    print_loss(preds, targets)


def train(patterns, output_dir, separate, val_size, num_epochs, batch_size, hid_units):
    train_data, val_data = get_dataset(patterns, val_size)
    model = WorkloadModel()
    model.init_model(hid_units, separate)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tmp_path = (Path(output_dir) / "tmp")
    tmp_path.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_path / "model.pt"

    target_model, target_idx = model.get_next_model()
    while target_model is not None:
        logger.info("Beginning to train model with target_idx: %s", target_idx)
        optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)
        process_train_data = model.prepare_inputs(train_data, target_idx=target_idx, train=True)
        train_data_loader = DataLoader(process_train_data, batch_size=batch_size)

        if val_data is not None:
            process_val_data = model.prepare_inputs(val_data, target_idx=target_idx, train=True)
            val_data_loader = DataLoader(process_val_data, batch_size=batch_size)

        # Begin training the model over the number of epochs.
        target_model.train()
        max_loss = None
        for epoch in range(num_epochs):
            loss_total = 0.

            for batch_idx, data_batch in enumerate(train_data_loader):
                variables = [Variable(var) for var in data_batch]

                optimizer.zero_grad()
                # We guarantee that the last element in the batch is the targets.
                outputs = target_model(*(variables[0:-1]))
                loss = loss_fn(outputs, variables[-1].float())
                loss_total += loss.item()
                loss.backward()
                optimizer.step()

            logger.info("Epoch %s, loss: %s", epoch, loss_total / len(train_data_loader))

            # Save the "best" by lowest loss.
            if max_loss is None or loss_total < max_loss:
                max_loss = loss_total
                torch.save(target_model.state_dict(), tmp_path)

        # Implant the "best" back into the state.
        state_dict = torch.load(tmp_path)
        target_model.load_state_dict(state_dict, strict=False)
        target_model, target_idx = model.get_next_model()

    model.save(output_dir)

    with torch.no_grad():
        process_train_data = model.prepare_inputs(train_data, target_idx=None, train=True)
        train_data_loader = DataLoader(process_train_data, batch_size=batch_size)

        # Get final training and validation set predictions
        logger.info("Training Loss Information:")
        predict(model, train_data_loader, separate)

        if val_data is not None:
            logger.info("Validation Loss Information:")
            process_val_data = model.prepare_inputs(val_data, target_idx=None, train=True)
            val_data_loader = DataLoader(process_val_data, batch_size=batch_size)
            predict(model, val_data_loader, separate)


class WorkloadTrainCLI(cli.Application):
    dir_input = cli.SwitchAttr(
        "--dir-input",
        str,
        mandatory=True,
        help="Path to the folder containing the input data.",
    )

    dir_output = cli.SwitchAttr(
        "--dir-output",
        str,
        mandatory=True,
        help="Path to the folder where output models should be written to.",
    )

    separate = cli.SwitchAttr(
        "--separate",
        str,
        help="Whether to train a separate model for each target or train a combined model",
    )

    val_size = cli.SwitchAttr(
        "--val-size",
        float,
        help="Size of the validation set to use.",
    )

    epochs = cli.SwitchAttr(
        "--epochs",
        int,
        help="Number of epochs to train the model for.",
    )

    batch_size = cli.SwitchAttr(
        "--batch-size",
        int,
        help="Size of the batch that should be used.",
    )

    hidden_size = cli.SwitchAttr(
        "--hidden-size",
        int,
        help="Number of hidden units to use.",
    )

    def main(self):
        train(self.dir_input, self.dir_output, self.separate == "True", self.val_size, self.epochs, self.batch_size, self.hidden_size)


if __name__ == "__main__":
    WorkloadTrainCLI.run()
