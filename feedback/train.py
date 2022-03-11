import copy
import json
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from sklearn.model_selection import KFold, ShuffleSplit

from .utils import set_seed
from .datasets import get_block_dataset, score, FeedbackDataset, max_labels
from .models import FeedbackModel, get_linear_warmup_power_decay_scheduler, split_batch
from .stats import EMAMeter, F1Meter, AverageMeter, MaxMeter, F1EMAMeter


def pretrain(fold, train_dataset, valid_dataset, config):
    print(f"Fold: {fold}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    train_dataset = get_block_dataset(
        train_dataset, "text", tokenizer, config["max_len"], config["seed"] + 4
    )
    valid_dataset = get_block_dataset(
        valid_dataset, "text", tokenizer, config["max_len"], config["seed"] + 4
    )
    print(f"Loading model: {config['model_path']}")
    model = AutoModelForMaskedLM.from_pretrained(config["model_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        config["lr"],
        betas=config["betas"],
        eps=config["eps"],
        weight_decay=config["wd"],
    )
    scheduler = get_linear_warmup_power_decay_scheduler(
        optimizer, config["warmup_steps"], float("inf"), power=-0.5
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config["mlm_prob"]
    )

    training_args = TrainingArguments(
        output_dir=f"./pre_{fold}",
        overwrite_output_dir=True,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps=config["eval_per_n_samples"],
        logging_steps=config["eval_per_n_samples"],
        save_steps=config["eval_per_n_samples"],
        save_total_limit=1,
        seed=config["seed"] + 3,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()
    shutil.rmtree(f"./pre_{fold}")
    trainer.save_model(f"./pre_{fold}")
    tokenizer.save_pretrained(f"./pre_{fold}")


def to_device(example, device):
    x, y, words, answer = example
    return x.to(device), y.to(device), words, answer


def forward(model, example):
    x, y, words, answer = example
    z = model(x)
    loss = model.get_loss(z, y, x)
    pred = model.get_pred(z, x)

    scores = score(pred, words, answer)
    return loss, scores


def forward_backward(model, example, model_batch_size, backward):
    x_batch, y_batch, words, answers = example
    x_chunks = split_batch(x_batch, model_batch_size)
    y_chunks = torch.split(y_batch, model_batch_size)
    z_chunks = []
    total_loss = []
    for x, y in zip(x_chunks, y_chunks):
        z = model(x)
        z_chunks.append(z.detach().clone())

        with amp.autocast():
            loss = model.get_loss(z, y, x)

        total_loss.append(loss.item())
        backward(loss)

    z_batch = torch.cat(z_chunks)

    pred = model.get_pred(z_batch, x_batch)
    scores = score(pred, words, answers)
    return np.average(total_loss), scores


def noop_backward(_):
    pass


def evaluate(model, device, data_loader, config):
    valid_loss_meter = AverageMeter()
    valid_f1_score = F1Meter()

    model.eval()
    with torch.no_grad():
        for example in data_loader:
            batch_size, example = example[0], example[1:]
            example = to_device(example, device)

            with amp.autocast():
                loss, scores = forward_backward(
                    model, example, config["model_batch_size"], noop_backward
                )

            valid_loss_meter.add(loss, batch_size)
            valid_f1_score.add(scores, batch_size)

    return valid_loss_meter.avg, valid_f1_score.f1


def train_loop(
    fold,
    model,
    device,
    optimizer,
    scheduler,
    scaler,
    train_loader,
    valid_loader,
    config,
):
    step_num = 0
    samples_since_eval = 0
    sample_num = 0

    train_loss_meter = EMAMeter(config["train_loss_ema"])
    train_f1_meter = F1EMAMeter(config["train_loss_ema"])
    best_meter = MaxMeter()

    def backward(loss):
        nonlocal step_num
        # Backward
        scaler.scale(loss / config["gradient_accumulation"]).backward()

        # Step
        if (step_num + 1) % config["gradient_accumulation"] == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        step_num += 1

    for epoch in range(config["num_epochs"]):
        print(f"Starting epoch {epoch}")
        for example in train_loader:
            batch_size, example = example[0], example[1:]
            example = to_device(example, device)

            # Forward
            loss, scores = forward_backward(
                model, example, config["model_batch_size"], backward
            )
            sample_num += batch_size
            samples_since_eval += batch_size

            # Stats
            train_loss_meter.add(loss, batch_size)
            train_f1_meter.add(scores, batch_size)

            # Validation
            if samples_since_eval >= config["eval_per_n_samples"]:
                samples_since_eval = 0
                valid_loss, valid_f1 = evaluate(model, device, valid_loader, config)
                is_best = best_meter.add(valid_f1)
                best = " (Best)" if is_best else ""

                results = [
                    f"epoch: {epoch:2d}",
                    f"sample: {sample_num:6d}",
                    f"lr: {optimizer.param_groups[0]['lr']:8.2e}",
                    f"train_loss: {train_loss_meter.avg:7.4f}",
                    f"valid_loss: {valid_loss:7.4f}",
                    f"train_error: {(train_f1_meter.f1):7.4f}",
                    f"valid_error: {(valid_f1):7.4f}",
                ]
                print(" | ".join(results) + best)
                if is_best:
                    torch.save(model.state_dict(), f"./best_{fold}.pth")
    return best_meter.max


def train(fold, train_dataset, valid_dataset, config):
    print(f"Fold: {fold}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    train_dataset = FeedbackDataset(
        train_dataset[0],
        train_dataset[1],
        tokenizer,
        config["max_len"],
        config["stride"],
        config["pad_to_multiple_of"],
    )
    valid_dataset = FeedbackDataset(
        valid_dataset[0],
        valid_dataset[1],
        tokenizer,
        config["max_len"],
        config["stride"],
        config["pad_to_multiple_of"],
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=4,
        batch_size=config["batch_size"],
        collate_fn=train_dataset.get_collate_fn(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        num_workers=4,
        batch_size=config["batch_size"],
        collate_fn=valid_dataset.get_collate_fn(),
    )

    print(f"Loading model: {config['model_path']}")
    model = FeedbackModel(
        config["model_path"], config["head"], max_labels, config["dropout"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        config["lr"],
        betas=config["betas"],
        eps=config["eps"],
        weight_decay=config["wd"],
    )
    scheduler = get_linear_warmup_power_decay_scheduler(
        optimizer, config["warmup_steps"], float("inf"), power=-0.5
    )
    scaler = amp.GradScaler()

    print(f"Training...")
    return train_loop(
        fold,
        model,
        device,
        optimizer,
        scheduler,
        scaler,
        train_loader,
        valid_loader,
        config,
    )


def get_dataset_splits_for_training(train_texts, train_df, config):
    datasets = []
    kf = ShuffleSplit(
        n_splits=config["folds"],
        test_size=config["mini_val"],
        random_state=config["seed"] + 1,
    )
    for train_index, valid_index in kf.split(train_texts):
        traind_t = train_texts.iloc[train_index].reset_index(drop=True)
        validd_t = train_texts.iloc[valid_index].reset_index(drop=True)
        traind_df = train_df[train_df["id"].isin(traind_t["id"])].reset_index(drop=True)
        validd_df = train_df[train_df["id"].isin(validd_t["id"])].reset_index(drop=True)
        datasets.append(((traind_t, traind_df), (validd_t, validd_df)))

    return datasets


def run(train_dfs, config, pre_config):
    config = copy.deepcopy(config)
    set_seed(config["seed"])
    print(f"Config: {json.dumps(config, indent=4, sort_keys=True)}")
    print(f"Pre-config: {json.dumps(pre_config, indent=4, sort_keys=True)}")
    datasets = get_dataset_splits_for_training(train_dfs[0], train_dfs[1], config)

    best_losses = []
    for fold, (train_dataset, valid_dataset) in enumerate(datasets):
        if pre_config is not None:
            pretrain(fold, train_dataset[0], valid_dataset[0], pre_config)
            config["model_path"] = f"./pre_{fold}"
        best_loss = train(fold, train_dataset, valid_dataset, config)
        best_losses.append(best_loss)
        if pre_config is not None:
            shutil.rmtree(f"./pre_{fold}")
    best_losses = np.array(best_losses)
    print(f"Best: {best_losses}")
    print(f"Avg cv: {np.average(best_losses)}")
