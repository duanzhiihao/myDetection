from typing import List
import torch
import torch.nn as nn

from utils.structures import ImageObjects


class SimpleBase(nn.Module):
    '''
    Base class for simple feature aggregation modules
    '''
    def __init__(self):
        super().__init__()
    
    def forward(self, features, hidden, is_start: torch.BoolTensor, additional=None):
        '''
        Forward pass with sanity check.

        Args:
            features: list of tensor. Current feature map.
            hidden: list of tensor or None. The previous hidden state.
            is_start: a batch of bool tensors indicating if the input x is the \
                start of a video.
        '''
        assert is_start.dim() == 1 and is_start.dtype == torch.bool

        if hidden is None:
            hidden = [torch.zeros_like(features[i]) for i in range(self.num_levels)]
        else:
            assert len(features) == len(hidden) == self.num_levels
            # if any image in the batch is a start of a video,
            # reset the corresponding hidden state
            if is_start.any():
                for level_hid in hidden:
                    assert level_hid.shape[0] == len(is_start)
                    level_hid[is_start].zero_()
        
        fused = self.fuse(features, hidden, additional)
        return fused

    def fuse(self, features, hidden, additional):
        raise NotImplementedError()


class WeightedAvg(SimpleBase):
    '''
    An example of weighted average feature aggregation.
    Too simple to be used in practice.
    '''
    def __init__(self, global_cfg: dict):
        super().__init__()
        self.num_levels = len(global_cfg['model.fpn.out_channels'])
        self.weights = nn.Parameter(torch.zeros(self.num_levels), requires_grad=True)

    def fuse(self, features, hidden, additional=None):
        raise NotImplementedError()
        out_feats = []
        for i in range(self.num_levels):
            cur = features[i]
            hid = hidden[i]

            w = torch.sigmoid(self.weights[i]) # weight
            fused = w*cur + (1-w)*hid # weighted average
            out_feats.append(fused)

        return out_feats


class Concat(nn.Module):
    '''
    Concatenate + convolutional layers
    '''
    def __init__(self, global_cfg: dict):
        super().__init__()
        from .modules import ConvBnLeaky
        fpn_out_chs = global_cfg['model.fpn.out_channels']
        self.num_levels = len(fpn_out_chs)
        self.rnns = nn.ModuleList()
        for ch in fpn_out_chs:
            fusion = nn.Sequential(
                nn.Conv2d(ch*2, ch, 1, stride=1, padding=0, groups=2),
                ConvBnLeaky(ch, ch//2, k=1, s=1),
                ConvBnLeaky(ch//2, ch, k=3, s=1)
            )
            self.rnns.append(fusion)

    def forward(self, features, hidden, is_start) -> List[torch.tensor]:
        assert is_start.dim() == 1 and is_start.dtype == torch.bool

        if hidden is None:
            hid_feat = [torch.zeros_like(features[i]) for i in range(self.num_levels)]
        else:
            hid_feat = hidden['fpn']
            assert len(features) == len(hid_feat) == self.num_levels
            # if any image in the batch is a start of a video,
            # reset the corresponding hidden state
            if is_start.any():
                for level_hid in hid_feat:
                    assert level_hid.shape[0] == len(is_start)
                    level_hid[is_start].zero_()

        out_feats = []
        for i, fusion in enumerate(self.rnns):
            cur: torch.tensor = features[i]
            hid: torch.tensor = hid_feat[i]
            assert cur.shape == hid.shape and cur.dim() == 4

            # concatenate and conv
            x = torch.cat([hid, cur], dim=1)
            x = fusion(x)
            assert x.shape == cur.shape
            out_feats.append(x)

        out_feats: List[torch.tensor]
        return out_feats


class CrossCorrelation(nn.Module):
    '''
    Previous detected features + cross-corelation
    '''
    def __init__(self, global_cfg: dict):
        super().__init__()
        from .modules import ConvBnLeaky
        fpn_out_chs = global_cfg['model.fpn.out_channels']
        self.num_levels = len(fpn_out_chs)
        # self.rnns = nn.ModuleList()
        # for ch in fpn_out_chs:
        #     fusion = nn.Sequential(
        #         ConvBnLeaky(ch, ch//2, k=1, s=1),
        #         ConvBnLeaky(ch//2, ch, k=3, s=1)
        #     )
        #     self.rnns.append(fusion)
        self.conf_thres = global_cfg['test.default_conf_thres']
        self.nms_thres  = global_cfg['test.nms_thres']

    def forward(self, features, hidden, is_start) -> List[torch.tensor]:
        '''
        Args:
            features: list of tensor, features
            hidden: dict
            is_start: torch.BoolTensor
        '''
        if hidden is None:
            return features
        
        # previous feature map and detections
        assert 'fpn' in hidden and 'final_pred' in hidden
        hid_feats = hidden['fpn']
        hid_preds = hidden['final_pred']

        out_feats = []
        for b in len(features[0].shape[0]): # batch
            _dts = hid_preds[b] # detections in an image
            _dts: ImageObjects
            _dts = _dts.post_process(self.conf_thres, self.nms_thres)

            raise NotImplementedError()

            for i in range(self.num_levels):
                cur = features[i]
                hid = hid_feats[i]
                assert cur.shape == hid.shape and cur.dim() == 4

        out_feats: List[torch.tensor]
        return out_feats


class FeatureBLockMatching(nn.Module):
    '''
    Disjoint block-matching in feature space
    '''
    def __init__(self, global_cfg: dict):
        super().__init__()
        from .modules import ConvBnLeaky
        fpn_out_chs = global_cfg['model.fpn.out_channels']
        self.num_levels = len(fpn_out_chs)
        
        self.matching_metric = global_cfg['model.agg.matching_metric']
        self.conf_thres = global_cfg['test.default_conf_thres']
        self.nms_thres  = global_cfg['test.nms_thres']

    def forward(self, features, hidden, is_start) -> List[torch.tensor]:
        '''
        Args:
            features: list of tensor, features
            hidden: dict
            is_start: torch.BoolTensor
        '''
        if hidden is None:
            return features
        
        # previous feature map and detections
        assert 'fpn' in hidden and 'final_pred' in hidden
        hid_feats = hidden['fpn']
        hid_preds = hidden['final_pred']

        out_feats = []
        for b in len(features[0].shape[0]): # batch
            _dts = hid_preds[b] # detections in an image
            _dts: ImageObjects
            _dts = _dts.post_process(self.conf_thres, self.nms_thres)

            raise NotImplementedError()

            for i in range(self.num_levels):
                cur = features[i]
                hid = hid_feats[i]
                assert cur.shape == hid.shape and cur.dim() == 4

        out_feats: List[torch.tensor]
        return out_feats
