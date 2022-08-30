from pathlib import Path
import pickle
import time
import numpy as np
import argparse
import pandas as pd
import glob
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Model Parameters
EPS = 1e-6
HISTOGRAM_LENGTH = 11
HISTOGRAM_POSITIONS = np.arange(0, 1.0 + 1.0 / (HISTOGRAM_LENGTH - 1), 1.0 / (HISTOGRAM_LENGTH - 1))

EMPTY_HIST_VALUE_FILL = -1.0
UNIQUE_HIST_VALUE_FILL = 1.0

MODEL_WORKLOAD_TARGETS = [
    "extend_percent",
    "defrag_percent",
    "hot_percent",
]

MODEL_WORKLOAD_NORMAL_INPUTS = [
    "free_percent",
    "dead_tuple_percent",
    "tuple_count",
    "tuple_len_avg",
    "target_ff",
    "select_dist",
    "insert_dist",
    "update_dist",
    "delete_dist",
]

MODEL_WORKLOAD_HIST_INPUTS = [
    ("data", "data_mlp1", "data_mlp2"),
    ("select", "select_mlp1", "select_mlp2"),
    ("insert", "insert_mlp1", "insert_mlp2"),
    ("update", "update_mlp1", "update_mlp2"),
    ("delete", "delete_mlp1", "delete_mlp2"),
]


class WorkloadModelInternal(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __init__(self, hid_units, num_outputs):
        super(WorkloadModelInternal, self).__init__()

        for (k, nn1, nn2) in MODEL_WORKLOAD_HIST_INPUTS:
            self[nn1] = nn.Linear(HISTOGRAM_LENGTH, hid_units)
            self[nn2] = nn.Linear(hid_units, hid_units)

        self.out_mlp1 = nn.Linear(hid_units * len(MODEL_WORKLOAD_HIST_INPUTS) + len(MODEL_WORKLOAD_NORMAL_INPUTS), hid_units)
        self.out_mlp2 = nn.Linear(hid_units, num_outputs)
        self.hid_units = hid_units
        self.num_outputs = num_outputs

    def state_dict(self):
        d = super(WorkloadModelInternal, self).state_dict()
        d["hid_units"] = self.hid_units
        d["num_outputs"] = self.num_outputs
        return d

    def forward(self, nonhist_inputs, data_dist, select_dist, insert_dist, update_dist, delete_dist,
                data_dist_mask, select_dist_mask, insert_dist_mask, update_dist_mask, delete_dist_mask):

        outputs = []
        for (key, nn1, nn2) in MODEL_WORKLOAD_HIST_INPUTS:
            s0 = self[nn1]
            s1 = self[nn2]
            data = locals()[f"{key}_dist"]
            mask = locals()[f"{key}_dist_mask"]

            output = F.relu(s0(data))
            output = F.relu(s1(output))
            output = output * mask
            output = torch.sum(output, dim=1, keepdim=False)
            norm = mask.sum(1, keepdim=False)
            output = output / norm
            outputs.append(output)

        outputs.append(nonhist_inputs)
        cat = torch.cat(tuple(o for o in outputs), 1)
        cat = F.relu(self.out_mlp1(cat))
        out = self.out_mlp2(cat)
        return out


class WorkloadModel:
    def save(self, output_dir):
        if self.model is not None:
            torch.save(self.model.state_dict(), Path(output_dir) / "combined_model.pt")
        else:
            for i, model in enumerate(self.models):
                torch.save(model.state_dict(), Path(output_dir) / f"model_target_{i}.pt")

    def load(self, output_dir):
        if (Path(output_dir) / "combined_model.pt").exists():
            f = Path(output_dir) / "combined_model.pt"
            state_dict = torch.load(f)
            hid_units = state_dict["hid_units"]
            num_outputs = state_dict["num_outputs"]
            self.model = WorkloadModelInternal(hid_units, num_outputs)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.models = []
            key = lambda x: int(x.split(".pt")[0].split("_")[-1])
            models = sorted(glob.glob(f"{output_dir}/model_target_*"), key=key)
            for model in models:
                state_dict = torch.load(model)
                hid_units = state_dict["hid_units"]
                num_outputs = state_dict["num_outputs"]

                model = WorkloadModelInternal(hid_units, num_outputs)
                model.load_state_dict(state_dict, strict=False)
                self.models.append(model)
            self.model = None

    def init_model(self, hid_units, separate):
        # Initialize all the models.
        if separate:
            self.models = []
            for _ in range(len(MODEL_WORKLOAD_TARGETS)):
                self.models.append(WorkloadModelInternal(hid_units, 1))
            self.next = 0
            self.model = None
        else:
            self.model = WorkloadModelInternal(hid_units, len(MODEL_WORKLOAD_TARGETS))
            self.next = 0

    def get_next_model(self):
        if self.model is not None:
            if self.next != 0:
                return None, None

            self.next = 1
            return self.model, None
        else:
            if self.next >= len(self.models):
                return None, None

            target = self.models[self.next]
            target_val = self.next
            self.next = self.next + 1
            return target, target_val

    @torch.no_grad()
    def predict(self, *args):
        args = [Variable(var) for var in args]
        if self.model is not None:
            self.model.eval()
            outputs = self.model(*args)
            preds = [o for o in outputs]
            return preds
        else:
            outputs = []
            for model in self.models:
                model.eval()
                output = model(*args)
                outputs.append(output)

            num_elem = len(outputs[0])
            assert num_elem > 0
            return [[o[i] for o in outputs] for i in range(num_elem)]

    def prepare_inputs(self, features, target_idx=None, train=True):
        nonhist_inputs = torch.tensor(features[MODEL_WORKLOAD_NORMAL_INPUTS].to_numpy(dtype=np.float32))
        if train:
            if target_idx is None:
                targets = torch.tensor(features[MODEL_WORKLOAD_TARGETS].to_numpy(dtype=np.float32))
            else:
                targets = torch.tensor(features[MODEL_WORKLOAD_TARGETS[target_idx]].to_numpy(dtype=np.float32))

        output_tensors = {}
        for (k, _, _) in MODEL_WORKLOAD_HIST_INPUTS:
            data = features[k].values
            max_block = np.max([len(record) for record in data]) + 1
            values = []
            masks = []

            for record in data:
                if len(record) == 0:
                    # Unfortunately, I think we do need to set something so we don't drive the set to nan.
                    # or collapse the tensor inappropriately. The (hope) insight is that it should learn that
                    # a 0 vector input means "nothing is useful"....
                    records = np.full((max_block, HISTOGRAM_LENGTH), EMPTY_HIST_VALUE_FILL)
                    mask = np.ones((max_block, 1))
                    values.append(records)
                    masks.append(mask)
                    continue

                # We need to pad everything so all the vectors within the batch-unit are of the same size.
                records = np.vstack(record)
                rows_to_pad = max_block - records.shape[0]
                mask = np.ones_like(records).mean(1, keepdims=True)
                assert records.shape[1] == HISTOGRAM_LENGTH
                records = np.pad(records, ((0, rows_to_pad), (0, 0)), mode="constant")
                mask = np.pad(mask, ((0, rows_to_pad), (0, 0)), 'constant')
                values.append(records)
                masks.append(mask)

            output_tensors[f"{k}_dist"] = torch.tensor(np.array(values, dtype=np.float32))
            output_tensors[f"{k}_dist_mask"] = torch.tensor(np.array(masks, dtype=np.float32))

        if train:
            return dataset.TensorDataset(nonhist_inputs,
                                         *[output_tensors[k] for k in output_tensors.keys() if k.endswith("_dist")],
                                         *[output_tensors[k] for k in output_tensors.keys() if k.endswith("_dist_mask")],
                                         targets)
        else:
            return dataset.TensorDataset(nonhist_inputs,
                                         *[output_tensors[k] for k in output_tensors.keys() if k.endswith("_dist")],
                                         *[output_tensors[k] for k in output_tensors.keys() if k.endswith("_dist_mask")])

    def featurize(queries, data, table_data, keyspace, train=True):
        num_insert = queries[queries.OP == "INSERT"].num_modify.sum()
        num_update = queries[queries.OP == "UPDATE"].num_modify.sum()
        num_delete = queries[queries.OP == "DELETE"].num_modify.sum()
        num_select_tuples = queries[queries.OP == "SELECT"].num_modify.sum()
        num_modify = num_insert + num_update + num_delete + num_select_tuples

        if train:
            # num_modify on SELECT is basically the number of tuples.
            # The defrag percent is computed with respect to the number of tuples.
            num_defrag = queries[queries.OP == "SELECT"].num_defrag.sum()
            num_extend = queries.num_extend.sum()
            num_hot = queries.num_hot.sum()
            assert num_update >= num_hot

        # Requires that if data is not valid, then there are no keys.
        assert data is not None or len(keyspace) == 0

        def kde_series(target):
            if len(target) == 0:
                # The case of no data.
                return [EMPTY_HIST_VALUE_FILL for x in range(len(HISTOGRAM_POSITIONS))]

            if target.nunique() == 1:
                # The case of only 1 unique value.
                return [UNIQUE_HIST_VALUE_FILL for x in range(len(HISTOGRAM_POSITIONS))]

            kernel = stats.gaussian_kde(data[key])
            return kernel(HISTOGRAM_POSITIONS)

        dists = {k: [] for (k, _, _) in MODEL_WORKLOAD_HIST_INPUTS}
        if len(keyspace) > 0:
            for key in keyspace:
                assert np.sum(data[key].isna()) == 0, print(f"{key} is somehow NA.")
                assert np.sum(queries[key].isna()) == 0, print(f"{key} is somehow NA in queries.")

                # Normalize data with respect to the entire dataset first.
                min_key = min(data[key].min(), queries[key].min())
                max_key = max(data[key].max(), queries[key].max())
                if max_key == min_key:
                    # Special case. Drive it all to 0.
                    max_key = max_key + 1

                queries[key] = (queries[key] - min_key) / (max_key - min_key)
                data[key] = (data[key] - min_key) / (max_key - min_key)
                assert np.sum(queries[key].isna()) == 0
                assert np.sum(data[key].isna()) == 0

            # Construct distribution for the general data.
            for key in keyspace:
                dists["data"].append(kde_series(data[key]))

            op = ["SELECT", "INSERT", "UPDATE", "DELETE"]
            for opcode in op:
                query_slice = queries[queries.OP == opcode]
                for key in keyspace:
                    output = kde_series(query_slice[key])
                    dists[opcode.lower()].append(output)

        if isinstance(table_data, dict):
            d = {
                "free_percent": table_data["approx_free_percent"],
                "dead_tuple_percent": table_data["dead_tuple_percent"],
                "tuple_count": table_data["approx_tuple_count"],
                "tuple_len_avg": table_data["tuple_len_avg"],
                "target_ff": table_data["ff"],
            }
        else:
            d = {
                "free_percent": table_data.approx_free_percent,
                "dead_tuple_percent": table_data.dead_tuple_percent,
                "tuple_count": table_data.approx_tuple_count,
                "tuple_len_avg": table_data.approx_tuple_len / table_data.approx_tuple_count,
                "target_ff": table_data.ff,
            }

        d["select_dist"] = 0.0 if num_modify == 0 else (num_select_tuples / num_modify)
        d["insert_dist"] = 0.0 if num_modify == 0 else (num_insert / num_modify)
        d["update_dist"] = 0.0 if num_modify == 0 else (num_update / num_modify)
        d["delete_dist"] = 0.0 if num_modify == 0 else (num_delete / num_modify)

        # Populate all the keys.
        for (k, _, _) in MODEL_WORKLOAD_HIST_INPUTS:
            d[k] = dists[k]

        if train:
            # Add the targets.
            d["extend_percent"] = 0.0 if (num_insert + num_update - num_hot) == 0 else num_extend / (num_insert + num_update - num_hot)
            d["defrag_percent"] = 0.0 if num_select_tuples == 0 else num_defrag / num_select_tuples
            d["hot_percent"] = 0.0 if num_update == 0 else num_hot / num_update

        return d
