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
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        self.entry_1 = wn(nn.Conv2d(3, 64, 3, 1, 1))

        body = nn.ModuleList()
        for i in range(self.n_blocks):
            body.append(FEB(wn, 64, 64))

        self.body = nn.Sequential(*body)

        self.reduction = BasicConv2d(wn, 64*13, 64, 1, 1, 0)

        self.upsample = UpsampleBlock(64, scale=scale, multi_scale=False, wn=wn, group=group)
        self.exit1 = wn(nn.Conv2d(64, 3, 3, 1, 1))


    def forward(self, x, scale):

        x = self.sub_mean(x)
        res = x
        x = self.entry_1(x)

        c0 = x
        out_blocks = []
        for i in range(self.n_blocks):
            x = self.body[i](x)
            out_blocks.append(x)

        out_blocks.append(c0)

        output = self.reduction(torch.cat(out_blocks, 1))

        output = output + x

        output = self.upsample(output, scale=scale)

        output = self.exit1(output)

        skip  = F.interpolate(res, (x.size(-2) * scale, x.size(-1) * scale), mode='bicubic', align_corners=False)

        output = skip + output

        output = self.add_mean(output)

        return output
