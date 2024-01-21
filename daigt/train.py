import copy
import json
import shutil
from types import SimpleNamespace

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

from sklearn.model_selection import KFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

from .utils import set_environ, set_seed
from .data import (
    get_block_dataset,
    DaigtDataset,
)
from .models import DaigtModel, get_linear_warmup_power_decay_scheduler, split_batch
from .stats import EMAMeter, AverageMeter, MaxMeter, TotalAucRocMeter, SlidingAucRocMeter

try:
    import bitsandbytes as bnb
except ImportError:
    pass


def register_optimizer(model, config):
    if config["optimizer"] in ["AdamW8bit"]:
        bnb.optim.GlobalOptimManager.get_instance().register_parameters(
            model.parameters()
        )


def get_optimizer(model, config):
    if config["optimizer"] == "AdamW":
        return optim.AdamW(
            model.parameters(),
            config["lr"],
            betas=config["betas"],
            eps=config["eps"],
            weight_decay=config["wd"],
        )
    elif config["optimizer"] == "AdamW8bit":
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            config["lr"],
            betas=config["betas"],
            eps=config["eps"],
            weight_decay=config["wd"],
        )

        embs = model.roberta.embeddings
        for emb_type in ["word", "position", "token_type"]:

            attr_name = f"{emb_type}_embeddings"

            if hasattr(embs, attr_name) and getattr(embs, attr_name) is not None:
                bnb.optim.GlobalOptimManager.get_instance().override_config(
                    getattr(embs, attr_name).weight, "optim_bits", 32
                )
        return optimizer
    else:
        raise RuntimeError("Unknown optimizer")


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
    register_optimizer(model, config)
    model = model.to(device)

    optimizer = get_optimizer(model, config)
    scheduler = get_linear_warmup_power_decay_scheduler(
        optimizer, config["warmup_steps"], float("inf"), power=-0.5
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config["mlm_prob"]
    )

    valid_batch_size = config["model_batch_size"] * config["valid_batch_multiplier"]
    training_args = TrainingArguments(
        output_dir=f"./pre_{fold}",
        overwrite_output_dir=True,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["model_batch_size"],
        per_device_eval_batch_size=valid_batch_size,
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
    x, y, labels = example
    return x.to(device), y.to(device), labels


def forward_backward(model: DaigtModel, example, model_batch_size, backward, autocast=True):
    x_batch, y_batch, labels = example
    x_chunks = split_batch(x_batch, model_batch_size)
    y_chunks = torch.split(y_batch, model_batch_size)
    z_chunks = []
    total_loss = []
    for x, y in zip(x_chunks, y_chunks):
        with amp.autocast(enabled=autocast):
            z = model(x)
            z_chunks.append(z.detach().clone())
            loss = model.get_loss(z, y, x)

        if loss is not None:
            total_loss.append(loss.item())
            backward(loss)

    z_batch = torch.cat(z_chunks)

    pred = model.get_pred(z_batch, x_batch)
    return np.average(total_loss), (pred, labels)


def noop_backward(_):
    pass


def evaluate(model, device, data_loader, config):
    valid_loss_meter = AverageMeter()
    valid_score_meter = TotalAucRocMeter()

    model.eval()
    valid_batch_size = config["model_batch_size"] * config["valid_batch_multiplier"]
    with torch.no_grad():
        for example in data_loader:
            batch_size, example = example[0], example[1:]
            example = to_device(example, device)

            loss, scores = forward_backward(
                model, example, valid_batch_size, noop_backward, config["autocast"]
            )

            valid_loss_meter.add(loss, batch_size)
            valid_score_meter.add(scores, batch_size)

    model.train()
    return valid_loss_meter.avg, valid_score_meter.auc


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
    wandb,
):
    step_num = 0
    samples_since_eval = 0
    sample_num = 0

    train_loss_meter = EMAMeter(config["train_loss_ema"])
    train_score_meter = SlidingAucRocMeter(config["train_loss_ema"])
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

    model.train()
    for epoch in range(config["num_epochs"]):
        print(f"Starting epoch {epoch}")
        for example in train_loader:
            batch_size, example = example[0], example[1:]
            example = to_device(example, device)

            # Forward
            loss, scores = forward_backward(
                model,
                example,
                config["model_batch_size"],
                backward,
                autocast=config["autocast"],
            )
            sample_num += batch_size
            samples_since_eval += batch_size

            # Stats
            train_loss_meter.add(loss, batch_size)
            train_score_meter.add(scores, batch_size)

            # Validation
            if samples_since_eval >= config["eval_per_n_samples"]:
                samples_since_eval = 0
                valid_loss, valid_score = evaluate(model, device, valid_loader, config)
                is_best = best_meter.add(valid_score)
                best = " (Best)" if is_best else ""

                results_dict = {
                    "epoch": epoch,
                    "sample": sample_num,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": train_loss_meter.avg,
                    "valid_loss": valid_loss,
                    "train_error": train_score_meter.auc,
                    "valid_error": valid_score,
                }
                results_format = {
                    "epoch": "{:2d}",
                    "sample": "{:6d}",
                    "lr": "{:8.2e}",
                    "train_loss": "{:7.4f}",
                    "valid_loss": "{:7.4f}",
                    "train_error": "{:7.4f}",
                    "valid_error": "{:7.4f}",
                }
                results = [
                    "{}: {}".format(key, value).format(results_dict[key])
                    for key, value in results_format.items()
                ]
                print(" | ".join(results) + best)
                wandb.log(results_dict)
                if is_best:
                    torch.save(model.state_dict(), f"./best_{fold}.pth")
    wandb.save("./best_{fold}.pth")
    return best_meter.max


def get_daigt_dataset(dataset, tokenizer, config, shuffle, valid):
    dataset = DaigtDataset(
        dataset,
        tokenizer,
        max_len=config["max_len"],
        stride=config["stride"],
        pad_to_multiple_of=config["pad_to_multiple_of"],
    )

    batch_size = config["batch_size"]
    if valid:
        batch_size *= config["valid_batch_multiplier"]
    loader = DataLoader(
        dataset,
        shuffle=shuffle,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=dataset.get_collate_fn(),
    )
    return dataset, loader


def train(fold, train_dataset, valid_dataset, config, wandb):
    print(f"Fold: {fold}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    _, train_loader = get_daigt_dataset(
        train_dataset, tokenizer, config, shuffle=True, valid=False
    )
    _, valid_loader = get_daigt_dataset(
        valid_dataset, tokenizer, config, shuffle=False, valid=True
    )

    print(f"Loading model: {config['model_path']}")
    model = DaigtModel(
        config["model_path"],
        config["head"],
        config["max_labels"],
        dropout=config.get("dropout"),
        weight=config.get("weight"),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    register_optimizer(model, config)
    model = model.to(device)
    model = torch.compile(model)

    optimizer = get_optimizer(model, config)
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
        wandb,
    )


def get_dataset_splits_for_training(dfs, config):
    all_train_dataset = dfs[config["train_dataset"]]
    all_valid_dataset = dfs[config["valid_dataset"]]

    datasets = []
    kf = StratifiedShuffleSplit(
        n_splits=config["folds"],
        test_size=config["mini_val"],
        random_state=config["kf_seed"],
    )
    for fold, (_, valid_index) in enumerate(kf.split(all_valid_dataset, all_valid_dataset["generated"])):
        valid_dataset = all_valid_dataset.iloc[valid_index].reset_index(drop=True)

        train_dataset = all_train_dataset[~all_train_dataset["source_id"].isin(all_valid_dataset["source_id"])]
        train_dataset = train_dataset.reset_index(drop=True)

        datasets.append((train_dataset, valid_dataset))
    return datasets


def run(dfs, config, pre_config, wandb):
    config = copy.deepcopy(config)
    wandb.config.update(config)
    wandb.config.update({"pre_config": pre_config})
    set_seed(config["seed"])
    set_environ()
    print(f"Config: {json.dumps(config, indent=4, sort_keys=True)}")
    print(f"Pre-config: {json.dumps(pre_config, indent=4, sort_keys=True)}")

    datasets = get_dataset_splits_for_training(dfs, config)

    best_scores = []
    for fold, (train_dataset, valid_dataset) in enumerate(datasets):
        if pre_config is not None:
            pretrain(fold, train_dataset, valid_dataset, pre_config)
            config["model_path"] = f"./pre_{fold}"
        best_score = train(fold, train_dataset, valid_dataset, config, wandb)
        best_scores.append(best_score)
        if pre_config is not None:
            shutil.rmtree(f"./pre_{fold}")
    best_scores = np.array(best_scores)
    cv_best = np.average(best_scores)
    print(f"Best: {best_scores}")
    print(f"Avg cv: {cv_best}")
    wandb.run.summary["cv_best"] = cv_best


class NoopWandB:
    def __init__(self):
        self.run = SimpleNamespace(summary={})
        self.config = {}

    def log(self, *_args, **_kwargs):
        pass

    def save(self, *_args, **_kwargs):
        pass
