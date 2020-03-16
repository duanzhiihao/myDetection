import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf

import models.backbones, models.fpns


class OneStageDetector(nn.Module):
    def __init__(self, backbone, fpn, head, **kwargs):
        super().__init__()
        self.backbone, info = get_backbone(backbone)
        self.fpn = get_fpn(fpn, info)
        self.head = get_head(head, **kwargs)
        
        self.input_normalization = kwargs.get('img_norm', False)

    def forward(self, x, labels=None):
        '''
        x: a batch of images, e.g. shape(8,3,608,608)
        labels: a batch of ground truth
        '''
        # normalization
        if self.input_normalization:
            for i in range(x.shape[0]):
                x[i] = tvf.normalize(x[i], [0.485,0.456,0.406], [0.229,0.224,0.225],
                                    inplace=True)
                # debug = (x.mean(), x.std())

        # go through the backbone and the feature payamid network
        features = self.backbone(x)
        features_fpn = self.fpn(features)
        dts, loss = self.head(features_fpn)

        if labels is None:
            return dts
        else:
            return loss


def get_backbone(name):
    if name == 'dark53':
        model = models.backbones.Darknet53()
        print("Using backbone Darknet-53. Loading ImageNet weights....")
        pretrained = torch.load('./weights/dark53_imgnet.pth')
        model.load_state_dict(pretrained)
        info = {'channels': (256, 512, 1024)}
        return model, info
    # elif name == 'res34':
    #     self.backbone = models.backbones.resnet34()
    # elif name == 'res50':
    #     self.backbone = models.backbones.resnet50()
    # elif name == 'res101':
    #     self.backbone = models.backbones.resnet101()
    # elif 'efficientnet' in backbone:
    #     self.backbone = models.backbones.efficientnet(backbone)
    else:
        raise Exception('Unknown backbone name')


def get_fpn(name, info):
    if name == 'yolo3':
        return models.fpns.YOLOv3FPN()
    else:
        raise Exception('Unknown backbone name')


def get_head(name, **kwargs):
    if name == 'yolo':
        from .yolov3 import YOLOLayer
        return YOLOLayer(1,1,1)
    else:
        raise Exception('Unknown backbone name')