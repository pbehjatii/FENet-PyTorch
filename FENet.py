import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import os
from ops import *


class FENet(nn.Module):

    def __init__(self, **kwargs):
        super(FENet, self).__init__()

        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.n_blocks = 12
        scale = kwargs.get("scale")
        group = kwargs.get("group", 4)

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)

        self.entry_1 = wn(nn.Conv2d(3, 64, 3, 1, 1))

        body = [FEB(wn, 64, 64) for _ in range(self.n_blocks)]
        self.body = nn.Sequential(*body)
        self.reduction = BasicConv2d(wn, 64*13, 64, 1, 1, 0)

        self.upscample = UpsampleBlock(64, scale=scale, multi_scale=False, wn=wn, group=group)
        self.exit = wn(nn.Conv2d(64, 3, 3, 1, 1))

        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)

    def forward(self, x, scale):

        x = self.sub_mean(x)
        res = x
        x = self.entry_1(x)

        c0 = x
        out_blocks = []

        out_blocks.append(c0)

        for i in range(self.n_blocks):

            x = self.body[i](x)
            out_blocks.append(x)

        output = self.reduction(torch.cat(out_blocks, 1))

        output = output + x

        output = self.upscample(output, scale=scale)
        output = self.exit(output)

        skip  = F.interpolate(res, (x.size(-2) * scale, x.size(-1) * scale), mode='bicubic', align_corners=False)

        output = skip + output

        output = self.add_mean(output)

        return output
