from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as sched

from transformers import AutoModel, BatchEncoding


class FF(nn.Module):
    def __init__(self, dim, ff_dim, output_dim):
        super().__init__()
        self.ff_linear = nn.Linear(dim, ff_dim)
        self.activation = F.gelu
        self.layer_norm = nn.LayerNorm(ff_dim)
        self.output = nn.Linear(ff_dim, output_dim)

    def forward(self, x):
        x = self.ff_linear(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.output(x)
        return x


class SoftmaxHead(nn.Module):
    def __init__(
        self,
        dim,
        ff_dim,
        output_dim,
        weight=None,
        ignore_idx=-1,
    ):
        super().__init__()
        self.ff = FF(dim, ff_dim, output_dim)
        weight = torch.tensor(weight, dtype=torch.float) if weight is not None else None
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_idx)
        self.ignore_idx = ignore_idx

    def forward(self, x, mask):
        x = x.select(dim=-2, index=0)
        x = self.ff(x)
        return x

    def get_loss(self, z, y, x):
        return self.loss(z, y)

    @staticmethod
    def get_pred(z, x):
        preds = F.softmax(z.detach(), dim=-1)
        preds = 1 - preds.select(dim=-1, index=0)
        to_sample = x.overflow_to_sample_mapping
        max_sample = to_sample.max().item() + 1
        sum_preds = torch.zeros(max_sample, dtype=preds.dtype, device=preds.device)
        sum_preds.scatter_add_(0, to_sample, preds)
        count_preds = torch.zeros(max_sample, dtype=preds.dtype, device=preds.device)
        count_preds.scatter_add_(0, to_sample, torch.ones_like(preds))
        avg_preds = sum_preds / count_preds
        return avg_preds.tolist()


class DaigtModel(nn.Module):
    def __init__(
        self,
        path,
        head,
        max_labels,
        dropout=None,
        weight=None,
        gradient_checkpointing=False,
    ):
        super().__init__()
        if dropout is None:
            self.roberta = AutoModel.from_pretrained(path)
        else:
            self.roberta = AutoModel.from_pretrained(
                path,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
            )
        if gradient_checkpointing:
            self.roberta.gradient_checkpointing_enable()

        config = self.roberta.config
        hidden_size = config.hidden_size
        if head == "softmax":
            self.head = SoftmaxHead(
                hidden_size,
                hidden_size,
                output_dim=max_labels,
                weight=weight,
            )
        else:
            raise RuntimeError("Unknown model head")

    def forward(self, x):
        mask = x.attention_mask
        x = {**x}
        x = {
            k: v
            for k, v in x.items()
            if k
            not in {
                "offset_mapping",
                "overflow_to_sample_mapping",
            }
        }
        x = self.roberta(**x)[0]
        x = self.head(x, mask)
        return x

    def get_loss(self, z, y, x):
        return self.head.get_loss(z, y, x)

    def get_pred(self, z, x):
        return self.head.get_pred(z, x)


def split_batch(x, size):
    x = {**x}
    items = []
    for k, v in x.items():
        v_split = torch.split(v, size)
        items.append(list(zip([k for _ in range(len(v_split))], v_split)))
    items = [BatchEncoding({k: v for k, v in batch}) for batch in zip(*items)]
    return items


def get_linear_warmup_power_decay_scheduler(
    optimizer, warmup_steps, max_num_steps, end_multiplier=0.0, power=1
):
    """Uses a power function a * x^power + b, such that it equals 1.0 at start_step=1
    and the end_multiplier at end_step. Afterwards, returns the end_multiplier forever.
    For the first warmup_steps, linearly increase the learning rate until it hits the power
    learning rate.
    """

    # a = end_lr - start_lr / (end_step ** power - start_step ** power)
    start_multiplier = 1.0
    start_step = 1
    scale = (end_multiplier - start_multiplier) / (
        max_num_steps**power - start_step**power
    )
    # b = start_lr - scale * start_step ** power
    constant = start_multiplier - scale * (start_step**power)

    def lr_lambda(step):
        step = start_step + step
        if step < warmup_steps:
            warmup_multiplier = scale * (warmup_steps**power) + constant
            return float(step) / float(max(1, warmup_steps)) * warmup_multiplier
        elif step >= max_num_steps:
            return end_multiplier
        else:
            return scale * (step**power) + constant

    return sched.LambdaLR(optimizer, lr_lambda)
