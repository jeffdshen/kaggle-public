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
