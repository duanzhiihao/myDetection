import torch


class RetinaLayer(torch.nn.Module):
    '''
    calculate the output boxes and losses
    '''
    def __init__(self, level_i: int, cfg: dict):
        super().__init__()

    def forward(self, raw: dict, img_size, labels=None):
        assert isinstance(raw, dict)