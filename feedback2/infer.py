import os
import gc

from tqdm import tqdm

import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

from .datasets import LABELS, Feedback2Dataset, MAX_LABELS
from .models import Feedback2Model, split_batch
from .train import to_device, get_feedback2_dataset


def get_preds(model, example, model_batch_size, autocast=True):
    x_batch, _, _ = example
    x_chunks = split_batch(x_batch, model_batch_size)
    z_chunks = []
    for x in x_chunks:
        with amp.autocast(enabled=autocast):
            z = model(x)
            z_chunks.append(z.detach().clone())
    z_batch = torch.cat(z_chunks)

    pred = model.get_pred(z_batch, x_batch)
    if "idxs" in x_batch:
        pred = list(zip(pred, x_batch.idxs.tolist()))
    return pred


def predict(dfs, path, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Feedback2Model(
        config["model_path"],
        config["head"],
        MAX_LABELS,
        dropout=config["dropout"],
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])

    test_dataset, test_loader = get_feedback2_dataset(
        (dfs["texts"], dfs["df"], None), tokenizer, config, shuffle=False, valid=True
    )

    predictions = []
    model.eval()
    valid_batch_size = config["model_batch_size"] * config["valid_batch_multiplier"]
    with torch.no_grad():
        for example in tqdm(test_loader):
            batch_size, example = example[0], example[1:]
            example = to_device(example, device)

            pred = get_preds(model, example, valid_batch_size)
            predictions += pred

    if config["head"] == "multi_token":
        predictions = {k: v for v, k in predictions}
        predictions = [predictions[idx] for idx in range(len(dfs["df"]))]
    return predictions


def predict_fold(dfs, config, fold):
    path = config["path"]
    file = config["file"]
    return predict(dfs, os.path.join(path, file.format(fold)), config)


def get_submission(dfs, preds_batch):
    preds_batch = np.array(preds_batch).tolist()
    sub = []
    df = dfs["df"]
    for idx, preds in zip(df["discourse_id"], preds_batch):
        sub.append([idx] + preds)

    sub = pd.DataFrame(sub, columns=["discourse_id"] + LABELS)
    return sub


def avg_ensemble(preds, weights):
    return np.average(preds, axis=0, weights=weights)

def log_avg_ensemble(preds, weights):
    preds = np.log(preds)
    avg_log = np.average(preds, axis=0, weights=weights)
    return np.exp(avg_log)

def ensemble(preds, weights, config):
    pred_weights = [(v, weights[k]) for k, v in preds.items()]
    preds, weights = zip(*pred_weights)
    preds = np.array(preds)
    weights = np.array(weights) / np.sum(weights)

    if config["prior"] is not None:
        preds += config["prior"]
        preds = preds / np.sum(preds, axis=-1, keepdims=True)

    if config["method"] == "avg":
        return avg_ensemble(preds, weights)
    elif config["method"] == "log_avg":
        return log_avg_ensemble(preds, weights)
    else:
        raise ValueError("No such method")