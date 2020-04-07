import json


def name_to_model(model_name):
    cfg = json.load(open(f'./configs/{model_name}.json', 'r'))

    if cfg['base'] == 'YOLO':
        from .yolov3 import YOLOv3
        model = YOLOv3(cfg)
    
    else:
        raise Exception('Unknown model name')
    
    return model, cfg

def deprecated_name_to_model(model_name):
    if model_name == 'test':
        from .yolov3 import YOLOv3
        cfg = {
            'backbone_fpn': 'b0_bifpn_345',
            'rpn': 'eff_w_conf',
            'num_anchor_per_level': 3,
            'pred_layer': 'YOLO',
            'num_class': 80,
            'input_format': 'RGB_1_norm',
        }
        return YOLOv3(cfg)
        
    elif model_name == 'yolov3':
        # darknet-53, YOLO fpn C3, 3x3 anchor boxes, 
        # xywh, norm by anchor, exp, xy:BCE, wh:L2
        from .yolov3 import YOLOv3
        cfg = {
            'backbone_fpn': 'dark53_yv3',
            'rpn': 'yv3',
            'num_anchor_per_level': 3,
            'pred_layer': 'YOLO',
            'num_class': 80,
            'input_format': 'RGB_1',
        }
        return YOLOv3(cfg)

    elif model_name == 'yv3_r50':
        # darknet-53, YOLO fpn C3, 3x3 anchor boxes, 
        # xywh, norm by anchor, exp, xy:BCE, wh:L2
        from .yolov3 import YOLOv3
        return YOLOv3(class_num=80, backbone='res50', img_norm=False)

    elif model_name == 'yv3_ltrb':
        # YOLOv3, only change the xywh to ltrb
        # darknet-53, fpn C3, 3x3 anchor boxes, ltrb, norm by anchor, exp, L2
        from .yolov3 import YOLOv3
        return YOLOv3(class_num=80, backbone='dark53', img_norm=False,
                      pred_layer='FCOS', ltrb='exp_l2')

    elif model_name == 'yv3_ltrb_sl1':
        # darknet-53, YOLO fpn C3, 3x3 anchor boxes, 
        # ltrb, norm by anchor, exp, smooth_L1
        from .yolov3 import YOLOv3
        return YOLOv3(class_num=80, backbone='dark53', img_norm=False,
                      pred_layer='FCOS', ltrb='exp_sl1')

    elif model_name == 'yv3_ltrb_RAG':
        # darknet-53, YOLO fpn C3, 3x3 anchor boxes, ltrb, norm by anchor, GIoU
        from .yolov3 import YOLOv3
        return YOLOv3(class_num=80, backbone='dark53', img_norm=False,
                      pred_layer='FCOS', ltrb='relu_ach_giou')
    
    # elif model_name == 'fcs_d53yc3_sl1':
    #     # darknet-53, YOLO fpn C3, 3 strides,
    #     # ltrb, norm by stride, exp, smooth_L1, center 1.5 stride, no centerness
    #     from .fcos import FCOS
    #     return FCOS(backbone='dark53', fpn='yolo3_1anch', ltrb_setting='exp_sl1',
    #                 img_norm=False)
        
    elif model_name == 'fcs2_yv3_expsl1':
        from .fcos2 import FCOS
        cfg = {
            'backbone_fpn': 'dark53_yv3',
            'rpn': 'yv3',
            'ltrb_setting': 'exp_sl1',
            'num_class': 80,
        }
        return FCOS(cfg)

    elif model_name == 'fcos_r50_fpn':
        from .fcos import FCOS
        cfg = {
            'backbone_fpn': 'res50_retina',
            'rpn': 'fcos',
            'ltrb_setting': 'relu_sl1',
            'num_class': 80,
        }
        return FCOS(cfg)
    
    elif model_name == 'fcs_r50_expsl1':
        from .fcos2 import FCOS
        cfg = {
            'backbone_fpn': 'res50_retina',
            'rpn': 'fcos',
            'ltrb_setting': 'exp_sl1',
            'num_class': 80,
        }
        return FCOS(cfg)

    elif model_name == 'd0_345_a3_conf_yolo':
        from .yolov3 import YOLOv3
        cfg = {
            'backbone_fpn': 'd0_345',
            'rpn': 'eff_w_conf',
            'head_repeat_num': 3,
            'num_anchor_per_level': 3,
            'pred_layer': 'YOLO',
            'num_class': 80,
            'input_format': 'RGB_1_norm',
        }
        return YOLOv3(cfg)
    
    elif model_name == 'd1_345_a3_conf_yolo':
        from .yolov3 import YOLOv3
        cfg = {
            'backbone_fpn': 'd1_345',
            'rpn': 'eff_w_conf',
            'head_repeat_num': 3,
            'num_anchor_per_level': 3,
            'pred_layer': 'YOLO',
            'num_class': 80,
            'input_format': 'RGB_1_norm',
        }
        return YOLOv3(cfg)
        
    elif model_name == 'd2_345_a3_conf_yolo':
        from .yolov3 import YOLOv3
        cfg = {
            'backbone_fpn': 'd2_345',
            'rpn': 'eff_w_conf',
            'head_repeat_num': 3,
            'num_anchor_per_level': 3,
            'pred_layer': 'YOLO',
            'num_class': 80,
            'input_format': 'RGB_1_norm',
        }
        return YOLOv3(cfg)

    elif model_name == 'eff-d0_yolo':
        from .effdet import EfficientDet
        cfg = {
            'model_id': 'd0',
            'C6C7': False,
            'pred_layer': 'YOLO',
            'num_anchor_per_level': 3,
            'wh_setting': 'exp_sl1',
            'num_class': 80,
            'input_format': 'RGB_1_norm',
        }
        return EfficientDet(cfg)
    
    # elif model_name in {'eff-d0'}:
    #     from .effdet import EfficientDet
    #     return EfficientDet(model_id=model_name[-2:], num_class=80)

    else:
        raise Exception('Unknown model name')
