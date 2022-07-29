from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as sched

from transformers import AutoModel, BatchEncoding

from .datasets import LABELS


def softmax(x, axis=None):
    x = x - np.max(x, axis=axis, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=axis, keepdims=True)


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


def softmax_pred(z, x):
    z = z.tolist()
    overflow_to_sample = x.overflow_to_sample_mapping.tolist()
    sample_to_scores = defaultdict(list)
    for i, m in enumerate(overflow_to_sample):
        sample_to_scores[m].append(z[i])

    preds = []
    for i in range(len(sample_to_scores)):
        scores = np.mean(sample_to_scores[i], axis=0)
        pred = softmax(scores, axis=-1)
        preds.append(pred)
    return preds


class ClassTokenHead(nn.Module):
    def __init__(
        self, dim, ff_dim, output_dim, bmpl_alpha=None, weight=None, ignore_idx=-1
    ):
        super().__init__()
        self.ff = FF(dim, ff_dim, output_dim)
        weight = torch.tensor(weight, dtype=torch.float) if weight is not None else None
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_idx)
        self.ignore_idx = ignore_idx
        self.bmpl_alpha = bmpl_alpha

    def forward(self, x, mask):
        x = self.ff(x[:, 0])
        return x

    def get_loss(self, z, y, x):
        if self.bmpl_alpha is None:
            return self.loss(z, y)

        if y.dim() < 2:
            return self.loss(z, y)

        lp = -F.log_softmax(z.detach(), dim=-1)
        p = F.softmax(z.detach(), dim=-1)
        lq = -torch.log(y)
        r = lp - lq
        r = torch.sigmoid((4 / self.bmpl_alpha) * r) * self.bmpl_alpha

        return self.loss(z, p * r)

    @staticmethod
    def get_pred(z, x):
        return softmax_pred(z, x)


class SiameseHead(nn.Module):
    def __init__(
        self, dim, ff_dim, output_dim, bmpl_alpha=None, weight=None, ignore_idx=-1
    ):
        super().__init__()
        self.ff = FF(dim, ff_dim, output_dim)
        weight = torch.tensor(weight, dtype=torch.float) if weight is not None else None
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_idx)
        self.ignore_idx = ignore_idx
        self.bmpl_alpha = bmpl_alpha

    def forward(self, x, mask):
        x = x[:, 0]
        x = x.view(2, -1, *x.shape[1:])
        a, b = torch.unbind(x)
        x = a + b
        x = self.ff(x)
        return x

    def get_loss(self, z, y, x):
        if self.bmpl_alpha is None:
            return self.loss(z, y)

        if y.dim() < 2:
            return self.loss(z, y)

        lp = -F.log_softmax(z.detach(), dim=-1)
        p = F.softmax(z.detach(), dim=-1)
        lq = -torch.log(y)
        r = lp - lq
        r = torch.sigmoid((4 / self.bmpl_alpha) * r) * self.bmpl_alpha

        return self.loss(z, p * r)

    @staticmethod
    def get_pred(z, x):
        return softmax_pred(z, x)


class Feedback2Model(nn.Module):
    def __init__(
        self,
        path,
        head,
        max_labels,
        dropout=None,
        weight=None,
        gradient_checkpointing=False,
        bmpl_alpha=1.0,
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
        if head == "class_token":
            self.head = ClassTokenHead(
                hidden_size,
                hidden_size,
                output_dim=max_labels,
                bmpl_alpha=bmpl_alpha,
                weight=weight,
            )
        elif head == "siamese":
            self.head = SiameseHead(
                hidden_size,
                hidden_size,
                bmpl_alpha=bmpl_alpha,
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
            if k not in {"offset_mapping", "overflow_to_sample_mapping"}
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
