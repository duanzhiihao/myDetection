import torch


def get_backbone(cfg: dict):
    backbone_name = cfg['model.backbone.name']
    if backbone_name == 'dark53':
        from .backbones import Darknet53
        assert cfg['model.backbone.num_levels'] == 3
        backbone = Darknet53()
        print("Using backbone Darknet-53. Loading ImageNet weights....")
        pretrained = torch.load('./weights/dark53_imgnet.pth')
        backbone.load_state_dict(pretrained)
        out_feature_channels = (256, 512, 1024)
        out_strides = (8, 16, 32)
    elif backbone_name.startswith('efficientnet'):
        from .backbones import EfNetBackbone
        backbone = EfNetBackbone(cfg)
        out_feature_channels = backbone.feature_chs
        out_strides = backbone.feature_strides
    else:
        raise Exception('Unknown backbone name')
    
    cfg['model.backbone.out_channels'] = out_feature_channels
    cfg['model.backbone.out_strides'] = out_strides
    return backbone


def get_fpn(cfg: dict):
    fpn_name = cfg['model.fpn.name']
    if fpn_name == 'yolov3':
        from .fpns import YOLOv3FPN
        fpn = YOLOv3FPN(cfg)
        out_feature_channels = cfg['model.backbone.out_channels']
        out_strides = cfg['model.backbone.out_strides']
    elif fpn_name == 'bifpn':
        from .fpns import get_bifpn
        fpn = get_bifpn(cfg)
        ch = cfg['model.bifpn.out_ch']
        out_feature_channels = [ch for _ in cfg['model.backbone.out_channels']]
        out_strides = cfg['model.backbone.out_strides']
    # elif name == 'retina':
    #     rpn = RetinaNetFPN(backbone_info['feature_channels'], 256)
    #     info = {
    #         'strides'
    #     }
    #     return rpn
    else:
        raise Exception('Unknown FPN name')

    cfg['model.fpn.out_channels'] = out_feature_channels
    cfg['model.fpn.out_strides'] = out_strides
    return fpn


def get_rpn(cfg: dict):
    rpn_name = cfg['model.rpn.name']
    if rpn_name == 'yolov3':
        from .rpns import YOLOHead
        rpn = YOLOHead(cfg)
    elif rpn_name == 'effrpn':
        from .rpns import EfDetHead
        rpn = EfDetHead(cfg)
    elif rpn_name == 'effrpn_ct':
        from .rpns import EfDetHead_wCenter
        rpn = EfDetHead_wCenter(cfg)
    else:
        raise NotImplementedError()
    return rpn


def get_det_layer(cfg: dict):
    det_layer_name = cfg['model.pred_layer']
    if det_layer_name == 'YOLO':
        from .detlayers.yolov3 import YOLOLayer
        return YOLOLayer
    elif det_layer_name == 'RetinaNet':
        from .detlayers.retinanet import RetinaLayer
        return RetinaLayer
    elif det_layer_name == 'FCOS':
        from .detlayers.fcos import FCOSLayer
        return FCOSLayer
    elif det_layer_name == 'FCOS2':
        from .detlayers.fcos2 import FCOSLayer
        return FCOSLayer
    elif det_layer_name == 'FCOS2_ATSS':
        from .detlayers.fcos2 import FCOS_ATSS_Layer
        return FCOS_ATSS_Layer
    elif det_layer_name == 'RAPiD':
        from .detlayers.rapid import RAPiDLayer
        return RAPiDLayer
    else:
        raise NotImplementedError()