from typing import List
import torch
import torch.nn as nn



class WeightedSum(nn.Module):
    def __init__(self, global_cfg: dict):
        super().__init__()
        self.num_levels = len(global_cfg['model.fpn.out_channels'])
        self.weights = nn.Parameter(torch.zeros(self.num_levels), requires_grad=True)

    def forward(self, features, hidden, is_start: torch.BoolTensor):
        '''
        Forward pass

        Args:
            features: list of tensor. Current feature map.
            hidden: list of tensor or None. The previous hidden state.
            is_start: a batch of bool tensors indicating if the input x is the \
                start of a video.
        '''
        if hidden is None:
            hidden = [torch.zeros_like(features[i]) for i in range(self.num_levels)]
        else:
            assert len(features) == len(hidden) == self.num_levels
            # if any image in the batch is a start of a video,
            # reset the corresponding hidden state
            if is_start.any():
                for level_hid in hidden:
                    level_hid: torch.tensor
                    level_hid[is_start].zero_()

        out_feats = []
        for i in range(self.num_levels):
            cur: torch.tensor = features[i]
            hid: torch.tensor = hidden[i]
            assert cur.shape == hid.shape
            w = torch.sigmoid(self.weights[i])
            # weighted sum
            fused = w*cur + (1-w)*hid
            out_feats.append(fused)

        out_feats: List[torch.tensor]
        return out_feats
