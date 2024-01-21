import os
import gc

from tqdm import tqdm

import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

from .data import DaigtDataset
from .models import DaigtModel, split_batch
from .train import to_device, get_daigt_dataset


def get_preds(model: DaigtModel, example, model_batch_size, autocast=True):
    x_batch, _, _ = example
    x_chunks = split_batch(x_batch, model_batch_size)
    z_chunks = []
    for x in x_chunks:
        with amp.autocast(enabled=autocast):
            z = model(x)
            z_chunks.append(z.detach().clone())
    z_batch = torch.cat(z_chunks)

    pred = model.get_pred(z_batch, x_batch)
    return pred


def predict(df, path, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DaigtModel(
        config["model_path"],
        config["head"],
        config["max_labels"],
        dropout=config.get("dropout"),
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])

    test_dataset, test_loader = get_daigt_dataset(
        df, tokenizer, config, shuffle=False, valid=True
    )

    predictions = []
    model.eval()
    valid_batch_size = config["model_batch_size"] * config["valid_batch_multiplier"]
    with torch.no_grad():
        for example in tqdm(test_loader):
            batch_size, example = example[0], example[1:]
            example = to_device(example, device)

            pred = get_preds(
                model, example, valid_batch_size, autocast=config["autocast"]
            )
            predictions += pred

    return predictions


def predict_fold(df, config, fold):
    path = config["path"]
    file = config["file"]
    return predict(df, os.path.join(path, file.format(fold)), config)


def get_submission(df, preds):
    sub = df[["id"]].copy(deep=True)
    sub["generated"] = preds
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
