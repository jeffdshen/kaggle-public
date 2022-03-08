from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as sched

from transformers import AutoModel, BatchEncoding

from .datasets import id_to_label


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
    def __init__(self, dim, ff_dim, output_dim):
        super().__init__()
        self.ff = FF(dim, ff_dim, output_dim)

    def forward(self, x, mask):
        x = self.ff(x)
        return x

    @staticmethod
    def get_loss(z, y, x, ignore_idx=-1):
        mask = x.attention_mask
        y = y.masked_fill(mask == 0, ignore_idx)
        return F.cross_entropy(
            z.transpose(1, -1), y.transpose(1, -1), ignore_index=ignore_idx
        )

    @staticmethod
    def get_pred(z, x):
        z = z.tolist()
        mask_batch = x.attention_mask.tolist()
        offsets_batch = x.offset_mapping.tolist()
        overflow_to_sample = x.overflow_to_sample_mapping.tolist()
        offset_to_scores = defaultdict(list)
        for i, offsets in enumerate(offsets_batch):
            m = overflow_to_sample[i]
            for j, offset in enumerate(offsets):
                if not mask_batch[i][j]:
                    continue
                offset_to_scores[(m, tuple(offset))].append(z[i][j])
        offset_to_argmax = {
            k: np.argmax(np.sum(v, axis=0)) for k, v in offset_to_scores.items()
        }

        pred = []
        prev_m = -1
        for i, offsets in enumerate(offsets_batch):
            m = overflow_to_sample[i]
            if m != prev_m:
                pred.append([])
                prev_m = m

            for j, offset in enumerate(offsets):
                if not mask_batch[i][j] or (offset[0] == offset[1] and offset[0] == 0):
                    continue

                label_id = offset_to_argmax.pop((m, tuple(offset)), None)
                if label_id is None:
                    continue
                label, label_start = id_to_label[label_id]

                # skip over, but don't invalidate the start label
                if label is None:
                    continue

                if label_start:
                    pred[-1].append([list(offset), label])
                    continue

                # Only valid with corresponding start label
                if len(pred[-1]) == 0 or pred[-1][-1][1] != label:
                    continue

                pred[-1][-1][0][1] = max(pred[-1][-1][0][1], offset[1])

        return pred


class FeedbackModel(nn.Module):
    def __init__(self, path, head, max_labels, dropout=None):
        super().__init__()
        if dropout is None:
            self.roberta = AutoModel.from_pretrained(path)
        else:
            self.roberta = AutoModel.from_pretrained(
                path,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
            )

        config = self.roberta.config
        hidden_size = config.hidden_size
        if head == "softmax":
            self.head = SoftmaxHead(hidden_size, hidden_size, output_dim=max_labels)
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
