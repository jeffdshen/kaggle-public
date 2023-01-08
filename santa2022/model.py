import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot2d(tensor, class_size):
    x, y = tensor.unbind(-1)
    x_hot = F.one_hot(x, class_size[0]).unsqueeze(-1)
    y_index = y.unsqueeze(-1).unsqueeze(-1).expand_as(x_hot)
    z = torch.zeros(*x.size(), *class_size, dtype=torch.long, device=tensor.device)
    z.scatter_(dim=-1, index=y_index, src=x_hot)
    return z


class ArmScalarEmbedding(nn.Module):
    def __init__(self, box_size, angle_sizes):
        super().__init__()
        self.box_size = box_size
        self.angle_sizes = nn.Parameter(torch.tensor(angle_sizes), requires_grad=False)

    def forward(self, arm):
        # auto broadcasts to n x a and promotes to default scalar type
        arm = arm / self.angle_sizes
        arm = arm.unsqueeze(-1).unsqueeze(-1).expand(*arm.size(), *self.box_size)
        arm = arm.transpose(-3, -2).transpose(-2, -1)
        return arm


class LocEmbedding(nn.Module):
    def __init__(self, box_size):
        super().__init__()
        self.box_size = box_size

    def forward(self, loc):
        return one_hot2d(loc, self.box_size).unsqueeze(-1)


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("Unsupported activation function: {}".format(activation))


class ObservationEmbedding(nn.Module):
    def __init__(self, box_size, angle_sizes, dim):
        super().__init__()
        self.arm_embed = ArmScalarEmbedding(box_size, angle_sizes)
        self.loc_embed = LocEmbedding(box_size)
        self.dim = dim

    def forward(self, colors, seen, arm, loc, target):
        seen = seen.unsqueeze(-1)
        arm = self.arm_embed(arm)
        loc = self.loc_embed(loc)
        target = self.loc_embed(target)

        x = torch.cat([colors, seen, arm, loc, target], dim=-1)
        pad = torch.zeros(
            *x.size()[:-1], self.dim - x.size(-1), dtype=x.dtype, device=x.device
        )
        x = torch.cat([x, pad], dim=-1)
        # H x W x C to C x H x W
        x = x.transpose(-1, -2).transpose(-2, -3).contiguous()
        return x


class DownsampleLayer(nn.Module):
    def __init__(self, in_dim, out_dim, factor):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, factor, stride=factor)

    def forward(self, x):
        return self.conv(x)


class BottleneckLayer(nn.Module):
    def __init__(self, dim, ff_dim, kernel_size, activation, num_groups):
        super().__init__()
        self.in_conv = nn.Conv2d(dim, ff_dim, 1, padding="same")
        self.ff_conv = nn.Conv2d(ff_dim, ff_dim, kernel_size, padding="same")
        self.out_conv = nn.Conv2d(ff_dim, dim, 1, padding="same")

        self.in_norm = nn.GroupNorm(num_groups, dim)
        self.ff_norm = nn.GroupNorm(num_groups, ff_dim)
        self.out_norm = nn.GroupNorm(num_groups, ff_dim)
        self.activation = get_activation_fn(activation)

    def forward(self, x):
        # Implements a bottleneck block, pre-activation style
        # if norm -> conv -> act (-> norm ...), then norm follows act, which is 0 or x
        # in transformers, between act and norm there is a residual, so x is more normal
        # if conv -> norm -> act, then act (nonnegative) is right next to residual
        # in resnet, this makes training more unstable (https://arxiv.org/abs/1603.05027)
        residual = x
        x = self.in_norm(x)
        x = self.activation(x)
        x = self.in_conv(x)

        x = self.ff_norm(x)
        x = self.activation(x)
        x = self.ff_conv(x)

        x = self.out_norm(x)
        x = self.activation(x)
        x = self.out_conv(x)
        x = residual + x

        return x


class ModelStage(nn.Module):
    def __init__(
        self,
        dim,
        in_dim,
        factor,
        depth,
        ff_dim,
        kernel_size,
        activation,
        num_groups,
    ):
        super().__init__()

        downsample_layer = DownsampleLayer
        layer = BottleneckLayer
        self.layers = nn.Sequential(
            downsample_layer(in_dim, dim, factor) if in_dim != dim else nn.Identity(),
            *[
                layer(dim, ff_dim, kernel_size, activation, num_groups)
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        return self.layers(x)


class FFHead(nn.Module):
    def __init__(self, dim, out_dim, activation):
        super().__init__()
        self.ff_linear = nn.Linear(dim, dim)
        self.activation = get_activation_fn(activation)
        self.layer_norm = nn.LayerNorm(dim)

        self.output = nn.Linear(dim, out_dim)

    def forward(self, x):
        x = self.ff_linear(x)
        x = self.activation(x)
        x = self.layer_norm(x)

        x = self.output(x)
        return x


class Model(nn.Module):
    def __init__(
        self,
        box_size,
        angle_sizes,
        action_size,
        stages,
        ff_dim_factor,
        kernel_size,
        activation,
        num_groups,
    ):
        super().__init__()
        self.embed = ObservationEmbedding(box_size, angle_sizes, stages[0][0])
        stage_list = [
            ModelStage(
                dim,
                in_dim,
                in_size // size,
                depth,
                dim // ff_dim_factor,
                kernel_size,
                activation,
                min(num_groups, dim // ff_dim_factor),
            )
            for (in_dim, in_size, _), (dim, size, depth) in zip(stages[:-1], stages[1:])
        ]
        self.encoder = nn.Sequential(*stage_list)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.head = FFHead(stages[-1][0], np.prod(action_size), activation)
        self.action_size = action_size

    # Inputs must be batched inputs (N, *) because of GroupNorm
    def forward(self, colors, seen, arm, loc, target):
        x = self.embed(colors, seen, arm, loc, target)
        x = self.encoder(x)
        x = self.pooler(x).squeeze(-1).squeeze(-1)
        x = self.head(x)
        x = x.view(*x.size()[:-1], *self.action_size)
        return x
