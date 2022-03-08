import os
import gc

import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

from .datasets import FeedbackDataset, pred_to_words, max_labels
from .models import FeedbackModel, split_batch
from .train import to_device


def get_preds(model, example, model_batch_size):
    x_batch, _, words, _ = example
    x_chunks = split_batch(x_batch, model_batch_size)
    z_chunks = []
    for x in x_chunks:
        z = model(x)
        z_chunks.append(z.detach().clone())
    z_batch = torch.cat(z_chunks)

    pred = model.get_pred(z_batch, x_batch)
    pred = pred_to_words(pred, words)

    return pred


def predict(
    texts,
    path,
    model_path,
    head,
    max_len,
    stride,
    batch_size,
    model_batch_size,
    device="cuda",
):
    model = FeedbackModel(model_path, head, max_labels)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    df = pd.DataFrame()
    test_dataset = FeedbackDataset(texts, df, tokenizer, max_len=max_len, stride=stride)
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        num_workers=4,
        batch_size=batch_size,
        collate_fn=test_dataset.get_collate_fn(),
    )

    predictions = []
    model.eval()
    with torch.no_grad():
        for example in test_loader:
            batch_size, example = example[0], example[1:]
            example = to_device(example, device)

            with amp.autocast():
                pred = get_preds(model, example, model_batch_size)
                predictions += pred

    del model, tokenizer, test_loader
    gc.collect()
    return predictions


def predict_fold(texts, model_path, train_config, fold):
    path = train_config.path
    file = train_config.file
    return predict(
        texts,
        os.path.join(path, file.format(fold)),
        model_path,
        head=train_config.head,
        max_len=train_config.max_len,
        stride=train_config.stride,
        batch_size=train_config.batch_size,
        model_batch_size=train_config.model_batch_size,
    )


def get_submission(texts, preds_batch):
    sub = []
    for idx, preds in zip(texts["id"], preds_batch):
        for words, label in preds:
            sub.append([idx, label, " ".join(str(word) for word in words)])

    sub = pd.DataFrame(sub, columns=["id", "class", "predictionstring"])
    return sub
